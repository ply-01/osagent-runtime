import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

type JsonObject = Record<string, unknown>;

interface MemoryInput {
	content: string;
	source_event_id: string;
	source_session_id: string;
}

interface CaseInput {
	case_id: string;
	question: string;
}

interface GroupInput {
	sample_id: string;
	memories: MemoryInput[];
	cases: CaseInput[];
}

interface EvaluationInput {
	top_k: number;
	retrieval_mode: "auto" | "fts";
	groups: GroupInput[];
}

interface RecallRow {
	id: string;
	content: string;
	metadata?: JsonObject | null;
	metadata_json?: string | null;
	[key: string]: unknown;
}

interface MnemopiInstance {
	remember(content: string, options?: JsonObject): string;
	recall(query: string, topK?: number, options?: JsonObject): Promise<RecallRow[]>;
	close(): void;
}

interface MnemopiConstructor {
	new (options?: JsonObject): MnemopiInstance;
}

function argumentsMap(args: string[]): Map<string, string> {
	const values = new Map<string, string>();
	for (let index = 0; index < args.length; index += 2) {
		const key = args[index];
		const value = args[index + 1];
		if (!key?.startsWith("--") || value === undefined) {
			throw new Error(`invalid argument near ${key ?? "<end>"}`);
		}
		values.set(key.slice(2), value);
	}
	return values;
}

function required(values: Map<string, string>, key: string): string {
	const value = values.get(key)?.trim();
	if (!value) throw new Error(`--${key} is required`);
	return value;
}

function safeFileName(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function metadataFor(row: RecallRow): JsonObject {
	if (row.metadata && typeof row.metadata === "object") return row.metadata;
	if (!row.metadata_json) return {};
	try {
		const parsed = JSON.parse(row.metadata_json);
		return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
	} catch {
		return {};
	}
}

async function main(): Promise<void> {
	const args = argumentsMap(process.argv.slice(2));
	const inputPath = resolve(required(args, "input"));
	const outputPath = resolve(required(args, "output"));
	const runtimeRoot = resolve(required(args, "runtime-root"));
	const dbRoot = resolve(required(args, "db-root"));
	const modulePath = join(
		runtimeRoot,
		"node_modules",
		"@oh-my-pi",
		"pi-mnemopi",
		"src",
		"index.ts",
	);
	const module = (await import(pathToFileURL(modulePath).href)) as {
		Mnemopi: MnemopiConstructor;
	};
	const input = JSON.parse(await readFile(inputPath, "utf8")) as EvaluationInput;
	await mkdir(dbRoot, { recursive: true });

	const groups: JsonObject[] = [];
	for (const group of input.groups) {
		const memory = new module.Mnemopi({
			dbPath: join(dbRoot, `${safeFileName(group.sample_id)}.sqlite3`),
			sessionId: group.sample_id,
			channelId: group.sample_id,
		});
		try {
			const indexStarted = performance.now();
			for (const item of group.memories) {
				memory.remember(item.content, {
					source: "locomo",
					importance: 0.5,
					metadata: {
						source_event_id: item.source_event_id,
						source_session_id: item.source_session_id,
					},
					extract: false,
					extractEntities: false,
					memoryType: "episode",
					veracity: "imported",
				});
			}
			const indexingMs = performance.now() - indexStarted;
			const cases: JsonObject[] = [];
			for (const item of group.cases) {
				const recallStarted = performance.now();
				const options = input.retrieval_mode === "fts" ? { queryEmbedding: null } : {};
				const hits = await memory.recall(item.question, input.top_k, options);
				const recallMs = performance.now() - recallStarted;
				cases.push({
					case_id: item.case_id,
					retrieved_event_ids: hits.map(row =>
						String(metadataFor(row).source_event_id ?? ""),
					),
					recall_ms: recallMs,
				});
			}
			groups.push({ sample_id: group.sample_id, indexing_ms: indexingMs, cases });
		} finally {
			memory.close();
		}
	}

	await mkdir(dirname(outputPath), { recursive: true });
	await writeFile(outputPath, `${JSON.stringify({ groups }, null, 2)}\n`, "utf8");
}

await main();
