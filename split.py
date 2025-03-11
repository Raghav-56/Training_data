#!/usr/bin/env python3
"""
Name: Data Splitter
Description: Splits files into training, validation, and test sets while preserving category structure.
Usage: python split.py <input_directory> <output_directory> [options]
"""

import os
import random
import shutil
import sys
import argparse
import logging
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm  # For progress bars

# Import the configured logger
from config.logger_config import logger


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


def split_files(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    file_extensions: Optional[List[str]] = None,
    copy_files: bool = False,
):
    """
    Split files into train, validation, and test sets while preserving category structure.
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Validate ratios
    test_ratio = 1 - train_ratio - val_ratio
    if (
        train_ratio <= 0
        or val_ratio < 0
        or test_ratio < 0
        or round(train_ratio + val_ratio + test_ratio, 10) != 1
    ):
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}. Must sum to 1."
        )

    # Validate directories
    in_dir = validate_directory(input_dir)
    out_dir = validate_directory(output_dir, create=True)

    # Create split directories
    train_dir = out_dir / "train"
    train_dir.mkdir(exist_ok=True)

    val_dir = None
    if val_ratio > 0:
        val_dir = out_dir / "val"
        val_dir.mkdir(exist_ok=True)

    test_dir = None
    if test_ratio > 0:
        test_dir = out_dir / "test"
        test_dir.mkdir(exist_ok=True)

    # Get category directories
    categories = [d for d in in_dir.iterdir() if d.is_dir()]
    if not categories:
        logger.warning(f"No category directories found in {in_dir}")
        return

    logger.info(f"Found {len(categories)} categories in input directory")

    # Process each category
    total_processed = 0

    for category in categories:
        category_name = category.name
        logger.info(f"Processing category: {category_name}")

        # Create corresponding directories
        train_category_dir = train_dir / category_name
        train_category_dir.mkdir(exist_ok=True)

        val_category_dir = None
        if val_dir:
            val_category_dir = val_dir / category_name
            val_category_dir.mkdir(exist_ok=True)

        test_category_dir = None
        if test_dir:
            test_category_dir = test_dir / category_name
            test_category_dir.mkdir(exist_ok=True)

        # Get files to process
        files = get_files_by_extensions(category, file_extensions)
        if not files:
            logger.warning(f"No matching files found in category: {category_name}")
            continue

        random.shuffle(files)

        # Calculate split indices
        train_idx = int(len(files) * train_ratio)
        val_idx = train_idx + int(len(files) * val_ratio)

        # Split files
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx] if val_ratio > 0 else []
        test_files = files[val_idx:] if test_ratio > 0 else []

        file_op = shutil.copy2 if copy_files else shutil.move

        # Process train files
        if train_files:
            for file in tqdm(train_files, desc=f"Train/{category_name}"):
                dest_file = train_category_dir / file.name
                file_op(file, dest_file)

        # Process validation files
        if val_files and val_category_dir:
            for file in tqdm(val_files, desc=f"Val/{category_name}"):
                dest_file = val_category_dir / file.name
                file_op(file, dest_file)

        # Process test files
        if test_files and test_category_dir:
            for file in tqdm(test_files, desc=f"Test/{category_name}"):
                dest_file = test_category_dir / file.name
                file_op(file, dest_file)

        total_processed += len(files)
        logger.info(
            f"Category {category_name} processed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test"
        )

    logger.info(f"Data splitting complete. Total files processed: {total_processed}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split files into train, validation, and test sets while preserving category structure."
    )

    parser.add_argument(
        "input_directory", help="Input directory containing category subdirectories"
    )
    parser.add_argument(
        "output_directory",
        help="Output directory where train/val/test splits will be created",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data for training set (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of data for validation set (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving them"
    )
    parser.add_argument(
        "--file-extensions",
        type=str,
        default=None,
        help="Comma-separated list of file extensions to include (e.g., mp4,avi)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Only show error messages")

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Configure log level based on arguments
    if args.verbose:
        logging.getLogger().handlers[0].setLevel(logging.DEBUG)  # Console handler
    elif args.quiet:
        logging.getLogger().handlers[0].setLevel(logging.ERROR)  # Console handler
    else:
        logging.getLogger().handlers[0].setLevel(logging.INFO)  # Console handler

    # Parse file extensions if provided
    extensions = None
    if args.file_extensions:
        extensions = args.file_extensions.split(",")
        logger.info(f"Filtering for file extensions: {extensions}")

    try:
        # Run the main function
        split_files(
            input_dir=args.input_directory,
            output_dir=args.output_directory,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
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
