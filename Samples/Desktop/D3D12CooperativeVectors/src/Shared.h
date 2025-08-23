#pragma once

// Things that can be shared between shaders and C++ code.
#if defined(__cplusplus)
typedef unsigned int uint;
#endif

static const uint IMAGE_WIDTH = 28;
static const uint IMAGE_HEIGHT = 28;
static const uint IMAGE_SIZE_IN_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;
static const uint IMAGE_SIZE_IN_PIXELS = IMAGE_SIZE_IN_BYTES;

static const uint NUM_INPUT_NEURONS = IMAGE_SIZE_IN_BYTES;
static const uint NUM_HIDDEN_NEURONS = 1;
static const uint NUM_OUTPUT_NEURONS = 10;