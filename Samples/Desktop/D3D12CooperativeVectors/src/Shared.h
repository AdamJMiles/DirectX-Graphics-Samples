#pragma once

// Things that can be shared between shaders and C++ code.
#if defined(__cplusplus)
typedef unsigned int uint;
#endif

#define RootSig "RootFlags(CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED), RootConstants(b0, num32BitConstants=3), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"

enum BindingSlots
{
    ROOT_CONSTANTS = 0,                 // b0
    IMAGE_BUFFER = 1,                   // t0
    LABEL_BUFFER = 2,                   // t1
    NETWORK_WEIGHTS_AND_BIASES_SRV = 3, // t2
    NETWORK_OFFSETS_SRV = 4,            // t3
    OUTPUT_OFFSETS_SRV = 5,             // t4
    NETWORK_WEIGHTS_AND_BIASES_UAV = 6, // u0
    FORWARD_OUTPUTS_UAV = 7,            // u1
	TEST_RESULTS_UAV = 7,               // u1 (same as above)
    DEBUG_UAV = 8,                      // u2
	ACCUMULATED_GRADIENTS_UAV = 9,      // u3
	EPOCH_RESULTS_UAV = 9			    // u3 (same as above)
};

static const uint IMAGE_WIDTH = 28;
static const uint IMAGE_HEIGHT = 28;
static const uint IMAGE_SIZE_IN_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT;
static const uint IMAGE_SIZE_IN_PIXELS = IMAGE_SIZE_IN_BYTES;

static const uint NUM_INPUT_NEURONS = IMAGE_SIZE_IN_BYTES;
static const uint NUM_HIDDEN_NEURONS = 32;
static const uint NUM_OUTPUT_NEURONS = 10;

struct NetworkOffsets
{
    uint weightsOffset;
    uint biasesOffset;
    uint accumulatedWeightsOffset;
	uint accumulatedBiasesOffset;
};