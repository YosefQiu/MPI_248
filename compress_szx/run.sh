#!/bin/bash

# Usage: ./run.sh <input_image> <min_x> <min_y> <max_x> <max_y> <output_base_name>

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <input_image> <min_x> <min_y> <max_x> <max_y> <output_base_name> <ground_truth_img>"
    exit 1
fi

INPUT_IMAGE=$1
MIN_X=$2
MIN_Y=$3
MAX_X=$4
MAX_Y=$5
OUTPUT_BASE_NAME=$6
GROUND_TRUTH_IMAGE=$7

# Process image to get RGB channels and save to binary file
CMD="python3 process_image.py 2 $INPUT_IMAGE $MIN_X $MIN_Y $MAX_X $MAX_Y"
echo "=== run === $CMD"
$CMD

# Define image dimensions
WIDTH=$(($MAX_X - $MIN_X))
HEIGHT=$(($MAX_Y - $MIN_Y))
CHANNEL_LENGTH=$(($WIDTH * $HEIGHT * 3))

# Perform compression and decompression with different error bounds
for ERROR_BOUND in 1E-1 1E-2 1E-3 1E-4 1E-5 1E-6 1E-7 1E-8 1E-9; do
    echo "======================================================================================="
    COMPRESSED_FILE=${OUTPUT_BASE_NAME}_rgb_channels_${ERROR_BOUND}.bin.szx
    DECOMPRESSED_FILE=${COMPRESSED_FILE}.out
    OUTPUT_IMAGE=${OUTPUT_BASE_NAME}_decompressed_${ERROR_BOUND}.png

    # Compress the binary file
    CMD="szx -z -f -i ./rgb_channels.bin -1 $CHANNEL_LENGTH -M ABS -A $ERROR_BOUND"
    echo "[=== run === $CMD]"
    $CMD

    CMD="mv ./rgb_channels.bin.szx $COMPRESSED_FILE"
    echo "[=== run === $CMD]"
    $CMD

    # Calculate and log the compression ratio
    CMD="python3 calculate_compression_ratio.py ./rgb_channels.bin $COMPRESSED_FILE ${OUTPUT_BASE_NAME}_CR.txt $ERROR_BOUND"
    echo "[=== run === $CMD]"
    $CMD

    # Decompress the binary file
    CMD="szx -x -f -s $COMPRESSED_FILE -1 $CHANNEL_LENGTH"
    echo "[=== run === $CMD]"
    $CMD

    # Reconstruct the image from the decompressed binary file
    CMD="python3 reconstruction.py $DECOMPRESSED_FILE $WIDTH $HEIGHT $OUTPUT_IMAGE"
    echo "[=== run === $CMD]"
    $CMD

    # 计算MSE PSNR bitrate
    CMD="python3 calculate_compression_metrics.py $GROUND_TRUTH_IMAGE $OUTPUT_IMAGE ${OUTPUT_BASE_NAME}_MPB.txt $ERROR_BOUND png"
    echo "[=== run === $CMD]"
    $CMD
done

echo "All operations completed."
