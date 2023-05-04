import os
import argparse
import shutil

def argParse():
    """Parses commandline args."""
    desc=''
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--header-file", default=None)
    parser.add_argument("--include-path", default=None)
    parser.add_argument("--build_dir")

    return parser.parse_args()

def main():
    args = argParse()

    # Copy to the include dir
    shutil.copyfile(args.header_file, os.path.join(args.build_dir, 'blocklib', args.include_path, os.path.basename(args.header_file)))                


if __name__ == "__main__":
    main()
