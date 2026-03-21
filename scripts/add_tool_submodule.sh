#!/usr/bin/env bash
#
# add_tool_submodule.sh
# 
# Helper script to add an external tool as a Git submodule.
# Ensures that URLs are correctly formatted and that all tools 
# are placed consistently within the `external/` directory.

# Exit immediately if a command exits with a non-zero status
set -e

# Default directory for external tools
EXTERNAL_DIR="external"

# Function to display usage
usage() {
    echo "Usage: $0 <repo_url> <tool_name>"
    echo ""
    echo "Arguments:"
    echo "  repo_url    The Git URL of the tool to add (e.g., https://github.com/wasserth/TotalSegmentator.git or huggingface.co/nvidia/NV-Segment-CTMR)"
    echo "  tool_name   The name of the tool (will create folder $EXTERNAL_DIR/<tool_name>)"
    echo ""
    echo "Examples:"
    echo "  $0 https://github.com/hhaentze/MRSegmentator.git MRSegmentator"
    echo "  $0 huggingface.co/nvidia/NV-Segment-CTMR NVSegmentCTMR"
    exit 1
}

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    usage
fi

REPO_URL=$1
TOOL_NAME=$2

# Fix URLs that don't start with http/https or git@
if [[ ! "$REPO_URL" =~ ^(https?://|git@) ]]; then
    echo "Fixing URL: prepending https:// to $REPO_URL"
    REPO_URL="https://$REPO_URL"
fi

TARGET_PATH="$EXTERNAL_DIR/$TOOL_NAME"

echo "============================================================"
echo "Adding new tool submodule..."
echo "Tool Name   : $TOOL_NAME"
echo "Repository  : $REPO_URL"
echo "Target Path : $TARGET_PATH"
echo "============================================================"

# Ensure Git repository is initialized
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Please run 'git init' first."
    exit 1
fi

# Create external directory if it doesn't exist
if [ ! -d "$EXTERNAL_DIR" ]; then
    mkdir -p "$EXTERNAL_DIR"
    echo "Created directory $EXTERNAL_DIR"
fi

# Check if target already exists to avoid submodule conflicts
if [ -d "$TARGET_PATH" ]; then
    echo "Warning: Target path $TARGET_PATH already exists!"
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# Add the submodule
echo "Running: git submodule add $REPO_URL $TARGET_PATH"
if git submodule add "$REPO_URL" "$TARGET_PATH"; then
    echo ""
    echo "✅ Successfully added $TOOL_NAME"
    echo "You can now commit this change: git commit -m \"Add $TOOL_NAME submodule\""
else
    echo ""
    echo "❌ Failed to add submodule. Check the URL and try again."
    exit 1
fi
