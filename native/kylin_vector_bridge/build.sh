#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-"$ROOT/libosagent_kylin_vector_bridge.so"}
SDK_INCLUDE=${OS_AGENT_KYLIN_VECTOR_INCLUDE:-/usr/include}
SDK_LIB=${OS_AGENT_KYLIN_VECTOR_LIBRARY_DIR:-/usr/lib/x86_64-linux-gnu}

if [[ ! -f "$SDK_INCLUDE/kysdk-vector-engine-client/Database.h" ]]; then
  echo "Kylin vector SDK development headers are missing: $SDK_INCLUDE/kysdk-vector-engine-client/Database.h" >&2
  echo "Install libkysdk-vector-engine-client-dev and retry." >&2
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

# Galaxy Kylin V11 images have shipped the same SDK headers with either the
# old or the C++11 libstdc++ string ABI. Probe both against the installed
# runtime instead of guessing from the package name. The output is accepted
# only when the dynamic loader reports no unresolved SDK symbols.
lib_dirs=("$SDK_LIB")
if command -v ldconfig >/dev/null 2>&1; then
  while IFS= read -r directory; do
    [[ -n "$directory" ]] && lib_dirs+=("$directory")
  done < <(ldconfig -p 2>/dev/null | awk '/libkysdk-vector-engine-client\.so/{print $NF}' | xargs -r -n1 dirname | sort -u)
fi

link_flags=()
rpath_flags=()
for directory in "${lib_dirs[@]}"; do
  link_flags+=("-L$directory")
  rpath_flags+=("-Wl,-rpath,$directory")
done

requested_abi=${OS_AGENT_KYLIN_CXX_ABI:-auto}
if [[ $requested_abi == auto ]]; then
  abi_candidates=(1 0)
else
  if [[ $requested_abi != 0 && $requested_abi != 1 ]]; then
    echo "OS_AGENT_KYLIN_CXX_ABI must be auto, 0, or 1" >&2
    exit 2
  fi
  abi_candidates=("$requested_abi")
fi

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
selected_abi=
for cxx_abi in "${abi_candidates[@]}"; do
  candidate="$build_dir/bridge-abi-${cxx_abi}.so"
  if ! g++ -std=c++17 -O2 -fPIC -shared \
      -D_GLIBCXX_USE_CXX11_ABI="$cxx_abi" \
      "$ROOT/kylin_vector_bridge.cpp" \
      -I"$SDK_INCLUDE" \
      "${link_flags[@]}" -Wl,-l:libkysdk-vector-engine-client.so.1 \
      "${rpath_flags[@]}" -Wl,-z,defs \
      -o "$candidate"; then
    continue
  fi
  loader_output=$(LD_LIBRARY_PATH=$(IFS=:; echo "${lib_dirs[*]}") ldd -r "$candidate" 2>&1 || true)
  if grep -Eq 'undefined symbol|not found' <<<"$loader_output"; then
    echo "candidate C++ ABI $cxx_abi is incompatible with the installed vector runtime:" >&2
    grep -E 'undefined symbol|not found' <<<"$loader_output" >&2 || true
    continue
  fi
  cp "$candidate" "$OUT"
  selected_abi=$cxx_abi
  break
done

if [[ -z $selected_abi ]]; then
  echo "Could not build a loadable Kylin vector bridge for the installed runtime." >&2
  echo "Install matching libkysdk-vector-engine-client and -dev packages, or set OS_AGENT_KYLIN_CXX_ABI." >&2
  exit 3
fi

printf '{"bridge_abi":2,"libstdcxx_cxx11_abi":%s,"sdk_include":"%s"}\n' \
  "$selected_abi" "$SDK_INCLUDE" >"$OUT.abi.json"
echo "built=$OUT cxx_abi=$selected_abi"
