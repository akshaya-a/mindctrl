#!/bin/bash

# Assert that the script is starting from the repository root
if [ ! -d "python" ] || [ ! -d "mindctrl-addon" ]; then
    echo "This script must be run from the root of the repository."
    exit 1
fi

# Define source and target directories
src_dir="python/src"
target_dir="mindctrl-addon/pysrc"

# Check if source directory exists
if [ ! -d "$src_dir" ]; then
    echo "Source directory $src_dir does not exist. Please check the path."
    exit 1
fi

# Check if target directory exists
if [ -d "$target_dir" ]; then
    echo "Target directory $target_dir already exists. Deleting it now..."
    rm -rf "$target_dir"
fi
# Create the target directory
echo "Creating target directory $target_dir..."
mkdir -p "$target_dir"

# Copy source directory to target directory
cp -r "$src_dir"/* "$target_dir"

echo "Python source files have been copied to $target_dir"

# Define source and target directories
src_dir="services"
target_dir="mindctrl-addon/services"

# Check if source directory exists
if [ ! -d "$src_dir" ]; then
    echo "Source directory $src_dir does not exist. Please check the path."
    exit 1
fi

# Check if target directory exists
if [ -d "$target_dir" ]; then
    echo "Target directory $target_dir already exists. Deleting it now..."
    rm -rf "$target_dir"
fi
# Create the target directory
echo "Creating target directory $target_dir..."
mkdir -p "$target_dir"

# Copy source directory to target directory
cp -r "$src_dir"/* "$target_dir"

echo "Services have been copied to $target_dir"
