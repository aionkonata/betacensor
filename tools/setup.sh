#!/usr/bin/env bash
#
# First-run setup: fetch ONNX Runtime, then build.
#
# The one thing a source drop cannot carry is ONNX Runtime -- the GPU build is
# 685 MB, and which build you want depends on the machine rather than on the
# program. So it is downloaded here, from Microsoft's own release tarballs, and
# unpacked into vendor/ where the Makefile finds it by wildcard.
#
# Everything else this script does is checking and reporting. It makes no
# system-wide changes, needs no root, installs nothing outside this directory,
# and can be run again safely.

set -euo pipefail

cd "$(dirname "$0")/.."

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- which build, and which version -----------------------------------------
#
# Two independent choices, and only one of them is about the GPU:
#
#   flavour  cpu or gpu. The GPU tarball is 685 MB against 17 MB and its only
#            advantage on a machine without an NVIDIA GPU is a provider that
#            will never load. Follows the hardware.
#
#   version  Which ONNX Runtime release. This is NOT a property of the GPU --
#            it is pinned from below by the models (an ORT refuses opsets newer
#            than it knows; everything in vendor/ is opset 12-20) and from
#            above, for the GPU build only, by which CUDA major the release was
#            compiled against. A GPU package built for CUDA 12 will not load
#            against a CUDA 11 runtime, and vice versa.
#
# Both can be overridden: `ORT_VERSION=1.19.2 ./tools/setup.sh gpu`.

# ORT releases and the CUDA major they are built against. From 1.19 the default
# GPU package targets CUDA 12; 1.18 and earlier target CUDA 11.
ORT_FOR_CUDA12="1.20.1"
ORT_FOR_CUDA11="1.18.1"
ORT_CPU_DEFAULT="1.20.1"
# Below this, releases predate the opset 20 the newer models here are exported
# at, and will refuse to load them.
ORT_MIN_USEFUL="1.16.0"

# The CUDA *runtime* major, which is what ORT dlopens -- not the toolkit
# version and not the driver's advertised maximum. Checked in that order of
# reliability: installed runtime library, then toolkit metadata, then the
# driver, which only gives an upper bound.
detect_cuda_major() {
	local m
	m=$(ldconfig -p 2>/dev/null | grep -o 'libcudart\.so\.[0-9]\+' \
	    | sed 's/.*\.//' | sort -rn | head -1)
	[ -n "$m" ] && { echo "$m"; return; }

	for d in /opt/cuda /usr/local/cuda; do
		if [ -r "$d/version.json" ]; then
			m=$(grep -o '"version"[^,]*' "$d/version.json" 2>/dev/null \
			    | head -1 | grep -o '[0-9]\+' | head -1)
			[ -n "$m" ] && { echo "$m"; return; }
		fi
	done

	if command -v nvidia-smi >/dev/null 2>&1; then
		m=$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: *[0-9]\+' \
		    | grep -o '[0-9]\+$' | head -1)
		[ -n "$m" ] && { echo "$m"; return; }
	fi
	echo ""
}

have_nvidia=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
	have_nvidia=1
fi

want="${1:-auto}"
[ "$want" = "auto" ] && { [ "$have_nvidia" = 1 ] && want="gpu" || want="cpu"; }

cuda_major="$(detect_cuda_major)"

if [ "$want" = "gpu" ]; then
	case "$cuda_major" in
	12)
		version="$ORT_FOR_CUDA12"
		say "CUDA 12 runtime detected -> ONNX Runtime $version (GPU)"
		;;
	11)
		version="$ORT_FOR_CUDA11"
		say "CUDA 11 runtime detected -> ONNX Runtime $version (GPU)"
		warn "This is the last release built against CUDA 11. Some models"
		warn "here are exported at opset 20 and may be refused by it; if a"
		warn "model fails to load, that is why. CUDA 12 is the better path."
		;;
	13)
		warn "CUDA 13 runtime detected. No ONNX Runtime release is built"
		warn "against it, and CUDA 13 dropped Pascal and older cards."
		warn "Falling back to the CPU build. To use the GPU, install a CUDA"
		warn "12 runtime alongside and re-run: ORT_VERSION=$ORT_FOR_CUDA12 $0 gpu"
		want="cpu"; version="$ORT_CPU_DEFAULT"
		;;
	"")
		if [ "$have_nvidia" = 1 ]; then
			warn "NVIDIA GPU present but no CUDA runtime found."
			warn "The GPU package needs libcudart and libcudnn at load time;"
			warn "without them it falls back to the CPU anyway, at 40x the"
			warn "download. Install your distro's CUDA runtime, or continue"
			warn "on CPU (what this will now do)."
		fi
		want="cpu"; version="$ORT_CPU_DEFAULT"
		;;
	*)
		warn "unrecognised CUDA major '$cuda_major' -- guessing the CUDA 12 build"
		version="$ORT_FOR_CUDA12"
		;;
	esac
else
	version="$ORT_CPU_DEFAULT"
	if [ "$have_nvidia" = 0 ]; then
		say "no NVIDIA GPU -> ONNX Runtime $version (CPU)"
		say "AMD note: ONNX Runtime's ROCm provider is not published as a"
		say "tarball and would have to be built from source. CPU inference is"
		say "the supported path here -- prefer a 320-input model over a 640."
	fi
fi

# An explicit ORT_VERSION beats all of the above, including the CUDA mapping:
# someone testing a specific release needs to be able to say so.
if [ -n "${ORT_VERSION:-}" ]; then
	say "ORT_VERSION=$ORT_VERSION overrides the detected $version"
	version="$ORT_VERSION"
fi

# A version older than the models is the one mistake worth refusing outright,
# because the failure it produces is "model fails to load" three steps later.
if [ "$(printf '%s\n%s\n' "$ORT_MIN_USEFUL" "$version" | sort -V | head -1)" != "$ORT_MIN_USEFUL" ]; then
	warn "ONNX Runtime $version predates $ORT_MIN_USEFUL and will refuse the"
	warn "newer models here (opset 20). Continuing, but expect load failures."
fi

ORT_VERSION="$version"
BASE="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"

case "$want" in
gpu) pkg="onnxruntime-linux-x64-gpu-${ORT_VERSION}" ;;
cpu) pkg="onnxruntime-linux-x64-${ORT_VERSION}" ;;
*)   die "usage: $0 [auto|cpu|gpu]   (ORT_VERSION=x.y.z overrides the version)" ;;
esac

say "ONNX Runtime: $want build ($pkg)"

# Any usable runtime already under vendor/ counts, whatever it is called. The
# tarball ships one as plain "vendor/onnxruntime", and matching only the
# versioned name meant a self-contained drop still went to the network on its
# first run -- for a copy of what it already had.
existing=""
for d in vendor/onnxruntime*; do
	[ -e "$d/lib/libonnxruntime.so" ] || continue
	if [ "$want" = "gpu" ]; then
		[ -e "$d/lib/libonnxruntime_providers_cuda.so" ] || continue
	fi
	existing="$d"
	break
done

if [ -n "$existing" ]; then
	say "already present: $existing -- skipping download"
elif [ -e "vendor/$pkg/lib/libonnxruntime.so" ]; then
	say "already present, skipping download"
else
	command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 \
		|| die "need curl or wget to download ONNX Runtime"

	mkdir -p vendor
	tarball="vendor/${pkg}.tgz"
	url="${BASE}/${pkg}.tgz"

	say "downloading $url"
	if command -v curl >/dev/null 2>&1; then
		curl -fL --progress-bar -o "$tarball" "$url"
	else
		wget -q --show-progress -O "$tarball" "$url"
	fi

	say "unpacking into vendor/"
	tar -xzf "$tarball" -C vendor/
	rm -f "$tarball"

	[ -e "vendor/$pkg/lib/libonnxruntime.so" ] \
		|| die "unpacked, but vendor/$pkg/lib/libonnxruntime.so is missing"
fi

# The CUDA execution provider needs cuDNN at *load* time. Its absence is not
# fatal -- the session falls back to CPU with a warning -- but silently getting
# CPU speed on a GPU machine is exactly the confusion worth pre-empting.
if [ "$want" = "gpu" ]; then
	if ! ldconfig -p 2>/dev/null | grep -q libcudnn; then
		warn "no libcudnn found on this system."
		warn "The CUDA provider will fail to load and inference will fall back"
		warn "to the CPU -- check for [CUDA] in the detector line at startup."
		warn "Install your distro's cuDNN package if you want GPU inference."
	fi
fi

# --- models -----------------------------------------------------------------
models=$(ls vendor/*.onnx 2>/dev/null | wc -l)
if [ "$models" -eq 0 ]; then
	warn "no .onnx models in vendor/ -- there is nothing to detect with."
	warn "Copy at least one detector model in, then re-run this script."
else
	say "$models model(s) in vendor/"
fi

# --- build ------------------------------------------------------------------
say "build configuration"
make config

say "building"
make -j"$(nproc 2>/dev/null || echo 4)"

say "checking this machine"
./tools/doctor.sh || true

cat <<'EOF'

Setup done. To run:

    ./build/betacensor live       # censor + settings panel
    ./build/betacensor doctor     # this machine's report, any time
    ./build/betacensor --help     # everything else

On KDE, read the window-rule section of PORTING.md before the first run --
without that rule the censoring flickers, and it looks like a bug in the
program rather than a missing setting.
EOF
