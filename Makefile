# BetaCensor-linux
#
# Deliberately a plain Makefile: no cmake, no generator. See DESIGN.md §4.
#
# Two spike binaries for now:
#   bc-capture  -- Phase 1: zero-copy screen capture into a GL texture
#   bc-overlay  -- Phase 2: click-through layer-shell overlay on one output
# Phase 3 joins them into a single process sharing one EGL context.

# --- Toolchain ----------------------------------------------------------------
# Nothing below may assume this machine. Everything hardware- or
# distro-specific is *detected* and falls back, because the build has to work
# on a tester's box with a different GPU vendor, a different libc and a
# different set of optional packages. `make config` prints what was found.
#
# Plain `CC ?= clang` cannot work: make predefines CC=cc, so ?= is a no-op.
# Only override when the value is make's built-in default, so `make CC=gcc`
# and an exported CC both still win. clang is preferred where present purely
# for build speed; gcc builds this fine.
ifeq ($(origin CC),default)
CC := $(if $(shell command -v clang 2>/dev/null),clang,cc)
endif
SCANNER ?= wayland-scanner

PKGS_COMMON  := egl gl libdrm
PKGS_OVERLAY := wayland-client wayland-egl

# sd-bus, for the ScreenCast portal. systemd is the usual provider, but the API
# is also carried by basu (systemd-free distros) and by elogind. Any of the
# three works; portal.c includes whichever header this picks. Without one of
# them there is no portal and so no capture, which is a hard stop rather than a
# degradation -- so it is reported by `make config` rather than left to a
# confusing compile error.
SDBUS_PKG := $(firstword $(foreach p,libsystemd basu libelogind,\
                 $(if $(shell pkg-config --exists $(p) && echo y),$(p))))
ifeq ($(SDBUS_PKG),basu)
SDBUS_DEF := -DBC_SDBUS_BASU=1
endif
ifeq ($(SDBUS_PKG),libelogind)
SDBUS_DEF := -DBC_SDBUS_ELOGIND=1
endif

PKGS_CAPTURE := libpipewire-0.3 $(SDBUS_PKG)
PKGS_ALL     := $(PKGS_COMMON) $(PKGS_CAPTURE) $(PKGS_OVERLAY)

# --- ONNX Runtime -------------------------------------------------------------
# Vendored rather than packaged (few distros carry it). Any directory matching
# vendor/onnxruntime* is a candidate, so a tester can drop in whatever release
# tarball suits their GPU without editing this file; a build carrying the CUDA
# provider wins over a plain CPU one, and ORT_DIR= on the command line beats
# both. Picking by wildcard rather than by pinned version is deliberate: the
# version in the name changes with every upstream release, and hardcoding it
# means the next person's download silently is not found.
ORT_CANDIDATES := $(sort $(wildcard $(abspath vendor/onnxruntime*)))
ORT_WITH_CUDA  := $(firstword $(foreach d,$(ORT_CANDIDATES),\
                      $(if $(wildcard $(d)/lib/libonnxruntime_providers_cuda.so),$(d))))
ORT_ANY        := $(firstword $(foreach d,$(ORT_CANDIDATES),\
                      $(if $(wildcard $(d)/lib/libonnxruntime.so),$(d))))
ORT_DIR ?= $(if $(ORT_WITH_CUDA),$(ORT_WITH_CUDA),$(ORT_ANY))
# A literal comma cannot appear inside a $(if ...) argument -- it would split
# the argument list -- and -Wl,-rpath, is full of them.
COMMA := ,
ORT_INC := $(if $(ORT_DIR),-I$(ORT_DIR)/include)
ORT_LIB := $(if $(ORT_DIR),-L$(ORT_DIR)/lib -lonnxruntime \
                           -Wl$(COMMA)-rpath$(COMMA)$(ORT_DIR)/lib)

# --- CUDA (optional) ----------------------------------------------------------
# Hands the preprocessed tensor to ONNX Runtime where it already lies instead of
# round-tripping it through host memory (DESIGN.md §0). Not a dependency:
# absent, BC_HAVE_CUDA is never defined and every path falls back to the
# readback -- which is the whole of what an AMD machine does, and it is a
# supported configuration rather than a broken one. Only libcudart is needed;
# there is no device code here, just the GL interop calls.
#
# Searched rather than pinned to /opt/cuda: Arch puts it there, Debian and
# Ubuntu use /usr/local/cuda, and Fedora installs into /usr.
# NO_CUDA=1 forces it off. Wanted on a machine that has the toolkit installed
# for something else but whose GPU is not NVIDIA: the interop calls would
# compile and then fail at runtime, and "build it out" is a clearer answer than
# a fallback path taken after an error.
CUDA_SEARCH := $(if $(NO_CUDA),,$(CUDA_DIR) /opt/cuda /usr/local/cuda /usr)
CUDA_FOUND  := $(firstword $(foreach d,$(CUDA_SEARCH),\
                   $(if $(wildcard $(d)/include/cuda_gl_interop.h),$(d))))
ifneq ($(CUDA_FOUND),)
# lib64 on most distros, lib on a few; take whichever holds the runtime.
CUDA_LIBDIR := $(firstword $(foreach l,lib64 lib,\
                   $(if $(wildcard $(CUDA_FOUND)/$(l)/libcudart.so*),$(CUDA_FOUND)/$(l))))
ifneq ($(CUDA_LIBDIR),)
CUDA_INC := -DBC_HAVE_CUDA=1 -I$(CUDA_FOUND)/include
CUDA_LIB := -L$(CUDA_LIBDIR) -lcudart -Wl,-rpath,$(CUDA_LIBDIR)
endif
endif

CFLAGS  += -std=c11 -O2 -g -Wall -Wextra -Wno-unused-parameter \
           -D_GNU_SOURCE -Ibuild/proto $(SDBUS_DEF) $(ORT_INC) $(CUDA_INC) \
           $(shell pkg-config --cflags $(PKGS_ALL))

# lld is materially faster than bfd and entirely optional. Probed rather than
# assumed: -fuse-ld=lld is a hard link error on a machine without it, which is
# a miserable first impression for someone who only wanted to try the thing.
ifneq ($(shell command -v ld.lld 2>/dev/null),)
LDFLAGS += -fuse-ld=lld
endif

LIBS_CAPTURE := $(shell pkg-config --libs $(PKGS_COMMON) $(PKGS_CAPTURE)) -lm
LIBS_OVERLAY := $(shell pkg-config --libs $(PKGS_COMMON) $(PKGS_OVERLAY)) -lm

# The first target in a Makefile is the default one, and the GUI rule below sits
# above `all:`. Without this, a bare `make` built only the GUI and left build/bc
# stale -- which looks exactly like an edit that did not take effect.
.DEFAULT_GOAL := all

# --- What did the build find? -------------------------------------------------
# The first thing to ask on a machine that is not this one, and the first thing
# to paste into a bug report. Deliberately prints the *decisions*, not the
# probes, because "CUDA: not found (CPU inference)" is a supported answer and
# reads as one, where a bare empty variable reads as a mistake.
.PHONY: config
config:
	@echo "compiler       : $(CC)"
	@echo "linker         : $(if $(findstring lld,$(LDFLAGS)),lld,system default)"
	@echo "sd-bus         : $(if $(SDBUS_PKG),$(SDBUS_PKG),NONE -- portal capture will not build)"
	@echo "ONNX Runtime   : $(if $(ORT_DIR),$(ORT_DIR),NONE -- see PORTING.md)"
	@echo "  CUDA provider: $(if $(wildcard $(ORT_DIR)/lib/libonnxruntime_providers_cuda.so),yes,no (CPU execution provider))"
	@echo "CUDA toolkit   : $(if $(CUDA_FOUND),$(CUDA_FOUND) [$(CUDA_LIBDIR)],not found (tensors go through host memory))"
	@echo "wayland-scanner: $(shell command -v $(SCANNER) 2>/dev/null || echo MISSING)"
	@echo "Qt6 Widgets    : $(if $(shell pkg-config --exists Qt6Widgets && echo y),yes,no -- the GUIs will not build)"
	@echo "ffmpeg         : $(shell command -v ffmpeg 2>/dev/null || echo 'not found (video mode only)')"

# Missing ONNX Runtime is the one failure worth catching before the compiler
# does: the error is otherwise a wall of missing headers, and the fix is a
# download rather than anything in the source.
ifeq ($(ORT_DIR),)
$(warning No ONNX Runtime found under vendor/onnxruntime*.)
$(warning Run ./tools/setup.sh, or set ORT_DIR=/path/to/onnxruntime.)
$(warning Only build/bc-capture and build/bc-overlay can build without it.)
endif

# --- GUI ---------------------------------------------------------------------
# One .cpp, one rule, no moc: gui.cpp declares no Q_OBJECT class, so every
# connection is a lambda and the build stays a single compile. See DESIGN.md §4.
QT_PKGS  := Qt6Widgets
# Probed, so `make` is the whole build on a machine with Qt and still succeeds
# on one without it. The panels are optional: the core censors without them,
# and a tester who only wants to know whether capture works should not be
# stopped by a missing Qt.
QT_OK    := $(shell pkg-config --exists Qt6Widgets 2>/dev/null && echo y)
QT_CFLAGS := $(shell pkg-config --cflags $(QT_PKGS)) -fPIC
QT_LIBS   := $(shell pkg-config --libs $(QT_PKGS))

build/betacensor-gui: src/gui.cpp | build
	$(CXX) -std=c++17 -O2 -g -Wall -Wextra $(QT_CFLAGS) -o $@ $< $(QT_LIBS)

build/bc-video-gui: src/video_gui.cpp | build
	$(CXX) -std=c++17 -O2 -g -Wall -Wextra $(QT_CFLAGS) -o $@ $< $(QT_LIBS)

build/bc-image-gui: src/image_gui.cpp | build
	$(CXX) -std=c++17 -O2 -g -Wall -Wextra $(QT_CFLAGS) -o $@ $< $(QT_LIBS)

gui: build/betacensor-gui build/bc-video-gui build/bc-image-gui

# --- Wayland protocol codegen -------------------------------------------------
# The layer-shell XML is not packaged on this system, so both protocols are
# vendored under vendor/. wayland-scanner turns them into C we compile directly;
# there is no libwayland-protocols build dependency.
PROTOS      := wlr-layer-shell-unstable-v1 xdg-shell
PROTO_HDRS  := $(patsubst %,build/proto/%-client-protocol.h,$(PROTOS))
PROTO_SRCS  := $(patsubst %,build/proto/%-protocol.c,$(PROTOS))
PROTO_OBJS  := $(patsubst build/proto/%.c,build/%.o,$(PROTO_SRCS))

# Without this make treats the generated .c as an intermediate and deletes it
# after linking, regenerating and recompiling it on every single build.
.SECONDARY: $(PROTO_SRCS) $(PROTO_HDRS)

build/proto/%-client-protocol.h: vendor/%.xml
	@mkdir -p $(dir $@)
	$(SCANNER) client-header $< $@

build/proto/%-protocol.c: vendor/%.xml
	@mkdir -p $(dir $@)
	$(SCANNER) private-code $< $@

build/%.o: build/proto/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# --- Binaries -----------------------------------------------------------------
CAPTURE_OBJ := build/main.o build/portal.o build/capture.o build/gpu.o
OVERLAY_OBJ := build/overlay_main.o build/overlay.o $(PROTO_OBJS)
BC_OBJ      := build/bc_main.o build/portal.o build/capture.o build/gpu.o \
               build/overlay.o build/preprocess.o build/detect.o build/config.o \
               build/track.o build/join.o build/seam.o build/landmark.o \
               build/hold.o build/failsafe.o $(PROTO_OBJS)

# The offline video pass. No capture and no Wayland: it renders headless into
# an FBO, so it links neither PipeWire nor the layer-shell protocol code.
VIDEO_OBJ   := build/video.o build/offline.o build/preprocess.o build/detect.o \
               build/config.o build/track.o build/join.o build/seam.o

# The offline image pass: the same headless half as the video pass, minus the
# tracker, plus the seam merge for sectioned images.
IMAGE_OBJ   := build/image.o build/offline.o build/preprocess.o build/detect.o \
               build/config.o build/join.o build/seam.o

BINS := build/bc-capture build/bc-overlay build/bc build/bc-video \
        build/bc-image build/betacensor

.PHONY: all clean capture overlay bc video image launcher
all: $(BINS) $(if $(QT_OK),build/betacensor-gui build/bc-video-gui build/bc-image-gui)
ifeq ($(QT_OK),)
$(warning Qt6Widgets not found -- building the core without the settings panels.)
endif
capture: build/bc-capture
overlay: build/bc-overlay
bc: build/bc
video: build/bc-video
image: build/bc-image
launcher: build/betacensor

build/bc-capture: $(CAPTURE_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS_CAPTURE)

build/bc-overlay: $(OVERLAY_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS_OVERLAY)

build/bc: $(BC_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS_CAPTURE) $(LIBS_OVERLAY) $(ORT_LIB) $(CUDA_LIB) -lpthread

# The launcher links nothing: it only starts the others and signals them.
build/betacensor: build/launcher.o
	$(CC) $(LDFLAGS) -o $@ $^

build/bc-video: $(VIDEO_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ -lEGL -lGL -lm $(ORT_LIB) $(CUDA_LIB) -lpthread

build/bc-image: $(IMAGE_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ -lEGL -lGL -lm $(ORT_LIB) $(CUDA_LIB) -lpthread

# Measurement tool: the detector's EYE class against the landmark model.
# Not part of `all`.
build/eyecmp.o: tools/eyecmp.c | $(PROTO_HDRS)
	$(CC) $(CFLAGS) -Isrc -MMD -MP -c -o $@ $<
build/bc-eyecmp: build/eyecmp.o build/offline.o build/preprocess.o \
                 build/detect.o build/landmark.o build/config.o
	$(CC) $(LDFLAGS) -o $@ $^ -lEGL -lGL -lm $(ORT_LIB) $(CUDA_LIB) -lpthread
eyecmp: build/bc-eyecmp
.PHONY: eyecmp

# Generated headers must exist before any .c that includes them is compiled.
#
# -MMD -MP emits a .d per object listing the headers it used, pulled back in
# below. Without it a header change rebuilt nothing: adding a field to struct
# bc_config left every stale .o addressing the *old* field offsets, so the
# binary linked and ran while disagreeing with itself about where cfg.censor
# was. That presents as a working pipeline whose detections never become boxes
# -- no crash, no warning, and nothing in the source to find. Cost is a few
# bytes of .d per object; the alternative is `make clean` being load-bearing.
build/%.o: src/%.c | $(PROTO_HDRS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

-include $(wildcard build/*.d)

# The GUI and test rules take this as an order-only prerequisite. Without a
# rule for it, `make clean && make test` (or `make gui`) stopped at "No rule to
# make target 'build'" -- it only ever worked because some earlier object rule
# had already created the directory.
build:
	@mkdir -p $@

clean:
	rm -rf build

# A source tarball for someone else's machine. See tools/mkdist.sh for what is
# included and what is deliberately left out.
.PHONY: dist
dist:
	@./tools/mkdist.sh $(DIST_ARGS)

# Fusion behaviour, measured without the screen: box-level jitter is dominated
# by whatever content is playing, so the mechanism is driven directly instead.
build/fusion_test: tests/fusion_test.c src/detect.c src/detect.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/fusion_test.c $(ORT_LIB) -lm

# The detector's outcome contract: a failed run is never a clean screen.
build/outcome_test: tests/outcome_test.c src/detect.c src/detect.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/outcome_test.c $(ORT_LIB) -lm -lpthread

# What the overlay may show: panic, the §6 deadlines, pause, reverse mode.
build/failsafe_test: tests/failsafe_test.c src/failsafe.c src/failsafe.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/failsafe_test.c -lm

build/track_test: tests/track_test.c src/track.c src/track.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/track_test.c -lm

build/join_test: tests/join_test.c src/join.c src/join.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/join_test.c -lm

build/seam_test: tests/seam_test.c src/seam.c src/seam.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/seam_test.c -lm

build/config_test: tests/config_test.c src/config.c src/config.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/config_test.c -lm

# The DMA-BUF modifier negotiation, driven without a compositor. Links
# capture.c whole (hence libpipewire) and stubs the GPU side, so it needs
# neither a display nor a GPU to run.
build/hold_test: tests/hold_test.c src/hold.c src/hold.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/hold_test.c src/hold.c -lm

build/capture_test: tests/capture_test.c src/capture.c src/capture.h | build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ tests/capture_test.c src/capture.c \
	    $(shell pkg-config --libs libpipewire-0.3) -lm

test: build/fusion_test build/outcome_test build/failsafe_test build/track_test build/join_test build/seam_test \
      build/config_test build/capture_test build/hold_test
	./build/fusion_test
	@echo
	./build/outcome_test
	@echo
	./build/failsafe_test
	@echo
	./build/track_test
	./build/join_test
	./build/seam_test
	./build/config_test
	@echo
	./build/capture_test
	@echo
	./build/hold_test
.PHONY: test
