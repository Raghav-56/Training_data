#!/usr/bin/env python3
"""
Name: Data Combiner
Description: Combines files from train, validation, and test sets into a single directory structure.
Usage: python combine.py <input_directory> <output_directory> [options]
"""

import os
import shutil
import sys
import argparse
import logging
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm  # For progress bars

# Import the configured logger
from config.logger_config import logger

# Import default settings
from config.settings import (
    COPY_FILES,
    FILE_EXTENSIONS,
    DEFAULT_OUTPUT_DIR,  # The directory where split data is stored (input for combine)
    DEFAULT_INPUT_DIR,  # The directory where original data is stored (output for combine)
)


def validate_directory(directory: str, create: bool = False) -> Path:
    """
    Validate that a directory exists or create it if specified.

    Args:
        directory: Directory path to validate
        create: If True, create the directory if it doesn't exist

    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        if create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        else:
            raise ValueError(f"Directory does not exist: {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")

    return dir_path


def get_files_by_extensions(
    directory: Path, extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Get list of files in directory with specific extensions.

    Args:
        directory: Directory to search for files
        extensions: List of file extensions to filter by (e.g., ['jpg', 'png'])

    Returns:
        List of Path objects for matching files
    """
    if not extensions:
        return [f for f in directory.iterdir() if f.is_file()]

    norm_extensions = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    ]
    return [
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in norm_extensions
    ]


def generate_unique_filename(base_path: Path, filename: str) -> Path:
    """
    Generate a unique filename if the original already exists.

    Args:
        base_path: Directory path where the file should be saved
        filename: Original filename

    Returns:
        Path with unique filename
    """
    dest_file = base_path / filename
    if not dest_file.exists():
        return dest_file

    # If file exists, add a counter to make it unique
    base, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{base}_{counter}{ext}"
        dest_file = base_path / new_filename
        if not dest_file.exists():
            return dest_file
        counter += 1


def combine_files(
    input_dir: str,
    output_dir: str,
    file_extensions: Optional[List[str]] = FILE_EXTENSIONS,
    copy_files: bool = COPY_FILES,  # Use the imported default
):
    """
    Combine files from train, validation, and test sets into a single directory structure.

    Args:
        input_dir: Directory containing train/val/test subdirectories
        output_dir: Directory where combined data will be placed
        file_extensions: List of file extensions to process
        copy_files: If True, copy files; if False, move files
    """
    # Validate directories
    in_dir = validate_directory(input_dir)
    out_dir = validate_directory(output_dir, create=True)

    # Look for train, val, test directories
    split_dirs = []
    for split_name in ["train", "val", "test"]:
        split_path = in_dir / split_name
        if split_path.exists() and split_path.is_dir():
            split_dirs.append((split_name, split_path))
        else:
            logger.warning(f"Split directory not found: {split_path}")

    if not split_dirs:
        logger.error(f"No train/val/test directories found in {in_dir}")
        return

    logger.info(
        f"Found {len(split_dirs)} split directories: {[d[0] for d in split_dirs]}"
    )

    # Process each split directory
    total_processed = 0

    # First, build a set of all unique categories across all splits
    all_categories = set()
    for split_name, split_path in split_dirs:
        categories = [d for d in split_path.iterdir() if d.is_dir()]
        for category in categories:
            all_categories.add(category.name)

    # Process each category
    for category_name in all_categories:
        logger.info(f"Processing category: {category_name}")

        # Create output category directory
        output_category_dir = out_dir / category_name
        output_category_dir.mkdir(exist_ok=True)

        # Process each split for this category
        for split_name, split_path in split_dirs:
            category_path = split_path / category_name
            if not category_path.exists() or not category_path.is_dir():
                logger.warning(f"Category {category_name} not found in {split_name}")
                continue

            # Get files to process
            files = get_files_by_extensions(category_path, file_extensions)
            if not files:
                logger.warning(
                    f"No matching files found in {split_name}/{category_name}"
                )
                continue

            # Copy or move files to output directory
            file_op = shutil.copy2 if copy_files else shutil.move
            for file in tqdm(files, desc=f"{split_name}/{category_name}"):
                # Generate a unique destination filename
                dest_file = generate_unique_filename(output_category_dir, file.name)
                file_op(file, dest_file)

            total_processed += len(files)
            logger.info(
                f"Processed {len(files)} files from {split_name}/{category_name}"
            )

    logger.info(f"Data combining complete. Total files processed: {total_processed}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine files from train, validation, and test sets into a single directory structure."
    )

    parser.add_argument(
        "--input",
        dest="input_directory",
        default=DEFAULT_OUTPUT_DIR,  # Corrected - use split output dir as input
        help="Input directory containing train/val/test subdirectories",
    )
    parser.add_argument(
        "--output",
        dest="output_directory",
        default=DEFAULT_INPUT_DIR,  # Corrected - use original data dir as output
        help="Output directory where combined data will be placed",
    )
    parser.add_argument(
        "--move",
        action="store_false",
        dest="copy",
        default=COPY_FILES,  # Use the imported default
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        default=None,
        help="Comma-separated list of file extensions to include (e.g., 'jpg,png,webp')",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Only show error messages")

    return parser.parse_args()


def setup_logging(verbose=False, quiet=False):
    """Configure logging based on verbosity settings."""
    log_level = logging.INFO  # Default log level
    if verbose:
        log_level = logging.DEBUG
    elif quiet:
        log_level = logging.ERROR

    # Configure root logger if not already configured
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    root_logger.addHandler(handler)

    # Set level for both the logger imported from config and the root logger
    logger.setLevel(log_level)
    root_logger.setLevel(log_level)


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Configure logging based on arguments
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Check if required arguments are provided
    if args.input_directory is None:
        logger.error(
            "Input directory is required. Use --input or set DEFAULT_OUTPUT_DIR in settings.py."
        )
        return 1

    if args.output_directory is None:
        logger.error(
            "Output directory is required. Use --output or set DEFAULT_INPUT_DIR in settings.py."
        )
        return 1

    # Parse file extensions if provided
    extensions = None
    if args.file_extensions:
        extensions = [ext.strip() for ext in args.file_extensions.split(",")]
        logger.info(f"Filtering for file extensions: {extensions}")

    try:
        # Run the main function
        combine_files(
            input_dir=args.input_directory,
            output_dir=args.output_directory,
            file_extensions=extensions,
            copy_files=args.copy,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            logger.debug(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
