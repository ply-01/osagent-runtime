// Thin C ABI wrapper around the official Kylin vector-engine C++ SDK.
// The Python runtime loads this library with ctypes; all SDK C++ types stay
// on the other side of the ABI boundary.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <kysdk-vector-engine-client/Database.h>

namespace {

using VectorDB::CollectionSchema;
using VectorDB::ConnectParam;
using VectorDB::DataType;
using VectorDB::Database;
using VectorDB::FieldDataPtr;
using VectorDB::FieldSchema;
using VectorDB::FloatVecFieldData;
using VectorDB::IndexDesc;
using VectorDB::IndexType;
using VectorDB::Int64FieldData;
using VectorDB::MetricType;
using VectorDB::SearchArguments;
using VectorDB::SearchResults;
using VectorDB::Status;
using VectorDB::VarCharFieldData;

constexpr const char* kIdField = "id";
constexpr const char* kVectorField = "vector";
constexpr const char* kKeyField = "memory_key";
constexpr std::uint32_t kBridgeAbiVersion = 2;

struct Handle {
    std::shared_ptr<Database> client;
    std::string collection;
    std::size_t dimension = 0;
    std::mutex mutex;
};

void set_error(char* buffer, std::size_t capacity, const std::string& message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    const std::size_t count = std::min(capacity - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

bool check_dimension(const Handle* handle, std::size_t dimension, char* error, std::size_t error_size) {
    if (dimension == 0 || dimension != handle->dimension) {
        std::ostringstream stream;
        stream << "vector dimension mismatch: expected " << handle->dimension << ", got " << dimension;
        set_error(error, error_size, stream.str());
        return false;
    }
    return true;
}

std::int64_t key_id(const char* key) {
    // FNV-1a gives a stable primary key while the original string remains the
    // returned VARCHAR field. A zero id is avoided because it is easy to
    // confuse with an uninitialised SDK value.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(key); *cursor != 0; ++cursor) {
        hash ^= *cursor;
        hash *= 1099511628211ULL;
    }
    hash |= 1ULL;
    return static_cast<std::int64_t>(hash);
}

Status ensure_collection(Handle& handle) {
    bool exists = false;
    auto status = handle.client->HasCollection(handle.collection, exists);
    if (!status.IsOk() && status.Code() != VectorDB::StatusCode::SERVER_FAILED) {
        return status;
    }
    if (exists) {
        return Status::OK();
    }

    CollectionSchema schema(handle.collection, "OSagent memory vectors");
    schema.AddField(FieldSchema(kIdField, DataType::INT64, "stable hashed memory key", true, false));
    schema.AddField(FieldSchema(kVectorField, DataType::FLOAT_VECTOR).WithDimension(static_cast<std::uint32_t>(handle.dimension)));
    schema.AddField(FieldSchema(kKeyField, DataType::VARCHAR).WithMaxLength(4096));
    // Use the default descriptor plus setters: Kylin SDK 1.2.0 has shipped
    // both a four-argument and a five-argument IndexDesc constructor. The
    // setter ABI is shared by those minor releases and avoids a runtime
    // undefined-symbol failure when the host package is one patch behind the
    // development headers used to build this bridge.
    IndexDesc index;
    index.SetFieldName(kVectorField);
    index.SetIndexName("osagent_memory_vector_index");
    index.SetIndexId(0);
    index.SetIndexType(IndexType::FLAT);
    index.SetMetricType(MetricType::COSINE);
    return handle.client->CreateCollection(schema, index);
}

template <typename Function>
int invoke(char* error, std::size_t error_size, Function&& function) {
    try {
        const auto status = function();
        if (status.IsOk()) {
            return 0;
        }
        set_error(error, error_size, status.Message());
        return static_cast<int>(status.Code());
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        return -1;
    } catch (...) {
        set_error(error, error_size, "unknown Kylin vector SDK exception");
        return -1;
    }
}

}  // namespace

extern "C" {

std::uint32_t osagent_kylin_vector_abi_version() { return kBridgeAbiVersion; }

int osagent_kylin_vector_key_id(const char* key, std::int64_t* result, char* error, std::size_t error_size) {
    if (key == nullptr || result == nullptr || *key == '\0') {
        set_error(error, error_size, "key and result are required");
        return -1;
    }
    *result = key_id(key);
    return 0;
}

int osagent_kylin_vector_open(
    const char* collection,
    std::size_t dimension,
    void** result,
    char* error,
    std::size_t error_size) {
    if (collection == nullptr || *collection == '\0' || result == nullptr || dimension == 0) {
        set_error(error, error_size, "collection, positive dimension, and result are required");
        return -1;
    }
    try {
        auto handle = std::make_unique<Handle>();
        handle->client = Database::Create();
        handle->collection = collection;
        handle->dimension = dimension;
        auto status = handle->client->Connect(ConnectParam("test"));
        if (!status.IsOk()) {
            set_error(error, error_size, status.Message());
            return static_cast<int>(status.Code());
        }
        status = ensure_collection(*handle);
        if (!status.IsOk()) {
            set_error(error, error_size, status.Message());
            return static_cast<int>(status.Code());
        }
        *result = handle.release();
        return 0;
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        return -1;
    } catch (...) {
        set_error(error, error_size, "unknown Kylin vector SDK exception");
        return -1;
    }
}

void osagent_kylin_vector_close(void* opaque) {
    if (opaque == nullptr) {
        return;
    }
    std::unique_ptr<Handle> handle(static_cast<Handle*>(opaque));
    std::lock_guard<std::mutex> lock(handle->mutex);
    if (handle->client) {
        handle->client->Disconnect();
    }
}

int osagent_kylin_vector_upsert(
    void* opaque,
    const char* const* keys,
    const float* vectors,
    std::size_t count,
    std::size_t dimension,
    char* error,
    std::size_t error_size) {
    if (opaque == nullptr || (count > 0 && (keys == nullptr || vectors == nullptr))) {
        set_error(error, error_size, "handle, keys, and vectors are required");
        return -1;
    }
    auto* handle = static_cast<Handle*>(opaque);
    if (!check_dimension(handle, dimension, error, error_size)) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    return invoke(error, error_size, [&]() {
        std::vector<std::int64_t> ids;
        std::vector<std::vector<float>> rows;
        std::vector<std::string> key_rows;
        ids.reserve(count);
        rows.reserve(count);
        key_rows.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            if (keys[row] == nullptr || *keys[row] == '\0') {
                return Status(VectorDB::StatusCode::INVALID_AGUMENT, "memory key must not be empty");
            }
            ids.push_back(key_id(keys[row]));
            rows.emplace_back(vectors + row * dimension, vectors + (row + 1) * dimension);
            key_rows.emplace_back(keys[row]);
        }
        if (count == 0) {
            return Status::OK();
        }
        std::vector<FieldDataPtr> fields{
            std::make_shared<Int64FieldData>(kIdField, std::move(ids)),
            std::make_shared<FloatVecFieldData>(kVectorField, std::move(rows)),
            std::make_shared<VarCharFieldData>(kKeyField, std::move(key_rows)),
        };
        VectorDB::DmlResults results;
        return handle->client->Upsert(handle->collection, fields, results);
    });
}

int osagent_kylin_vector_delete(
    void* opaque,
    const char* const* keys,
    std::size_t count,
    char* error,
    std::size_t error_size) {
    if (opaque == nullptr || (count > 0 && keys == nullptr)) {
        set_error(error, error_size, "handle and keys are required");
        return -1;
    }
    auto* handle = static_cast<Handle*>(opaque);
    std::lock_guard<std::mutex> lock(handle->mutex);
    return invoke(error, error_size, [&]() {
        if (count == 0) {
            return Status::OK();
        }
        std::ostringstream expression;
        expression << kIdField << " in [";
        for (std::size_t index = 0; index < count; ++index) {
            if (keys[index] == nullptr || *keys[index] == '\0') {
                return Status(VectorDB::StatusCode::INVALID_AGUMENT, "memory key must not be empty");
            }
            if (index != 0) {
                expression << ",";
            }
            expression << key_id(keys[index]);
        }
        expression << "]";
        VectorDB::DmlResults results;
        return handle->client->Delete(handle->collection, expression.str(), results);
    });
}

int osagent_kylin_vector_search(
    void* opaque,
    const float* query,
    std::size_t dimension,
    std::size_t top_k,
    const char* const* allowed_keys,
    std::size_t allowed_count,
    char** output,
    std::size_t* output_size,
    char* error,
    std::size_t error_size) {
    if (opaque == nullptr || query == nullptr || output == nullptr || output_size == nullptr || top_k == 0 ||
        (allowed_count > 0 && allowed_keys == nullptr)) {
        set_error(error, error_size, "handle, query, positive top_k, output, and output_size are required");
        return -1;
    }
    auto* handle = static_cast<Handle*>(opaque);
    if (!check_dimension(handle, dimension, error, error_size)) {
        return -1;
    }
    *output = nullptr;
    *output_size = 0;
    std::lock_guard<std::mutex> lock(handle->mutex);
    try {
        SearchArguments arguments(handle->collection, static_cast<std::int64_t>(top_k), MetricType::COSINE);
        arguments.AddOutputField(kKeyField);
        arguments.SetGuaranteeTimestamp(VectorDB::GuaranteeStrongTs());
        if (allowed_count > 0) {
            std::ostringstream expression;
            expression << kIdField << " in [";
            for (std::size_t index = 0; index < allowed_count; ++index) {
                if (allowed_keys[index] == nullptr || *allowed_keys[index] == '\0') {
                    set_error(error, error_size, "allowed memory key must not be empty");
                    return -1;
                }
                if (index != 0) {
                    expression << ",";
                }
                expression << key_id(allowed_keys[index]);
            }
            expression << "]";
            auto status = arguments.SetExpression(expression.str());
            if (!status.IsOk()) {
                set_error(error, error_size, status.Message());
                return static_cast<int>(status.Code());
            }
        }
        arguments.AddTargetVector(kVectorField, std::vector<float>(query, query + dimension));
        SearchResults results;
        auto status = handle->client->Search(arguments, results);
        if (!status.IsOk()) {
            set_error(error, error_size, status.Message());
            return static_cast<int>(status.Code());
        }
        std::ostringstream stream;
        if (!results.Results().empty()) {
            auto& result = results.Results().front();
            const auto& ids = result.Ids().IntIDArray();
            const auto& scores = result.Scores();
            auto field = result.OutputField(kKeyField);
            auto key_field = std::static_pointer_cast<VarCharFieldData>(field);
            const std::vector<std::string> empty_keys;
            const auto& returned_keys = key_field ? key_field->Data() : empty_keys;
            const std::size_t rows = std::min(ids.size(), scores.size());
            for (std::size_t index = 0; index < rows; ++index) {
                stream << ids[index] << '\t' << scores[index];
                if (index < returned_keys.size()) {
                    stream << '\t' << returned_keys[index];
                }
                stream << '\n';
            }
        }
        const auto text = stream.str();
        auto* buffer = static_cast<char*>(std::malloc(text.size() + 1));
        if (buffer == nullptr) {
            set_error(error, error_size, "unable to allocate search result buffer");
            return -1;
        }
        std::memcpy(buffer, text.data(), text.size());
        buffer[text.size()] = '\0';
        *output = buffer;
        *output_size = text.size();
        return 0;
    } catch (const std::exception& exception) {
        set_error(error, error_size, exception.what());
        return -1;
    } catch (...) {
        set_error(error, error_size, "unknown Kylin vector SDK exception");
        return -1;
    }
}

void osagent_kylin_vector_free(void* buffer) { std::free(buffer); }

}  // extern "C"
