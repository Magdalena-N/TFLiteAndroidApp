import argparse
import os
import requests
import urllib.request


def get_dataset_from_file(file_path, out_dir):
    dataset_links = []
    with open(file_path) as dataset_file:
        for line in dataset_file:
            label = line.split(" ")[0]
            dataset_links.append(line.split(" ")[1])
            get_dataset_from_links(dataset_links, out_dir + os.path.sep + label)


def get_dataset_from_links(links, out_dir):
    for link in links:
        with urllib.request.urlopen(link) as page:
            page_content = page.read()
            page_content = page_content.split(b'\r\n')
            for page in page_content:
                # need to get rid of b prefix from casting byte to string
                page = str(page)[1:].replace("'", "")
                try:
                    if __if_exists(page):
                        print("Downloading %s" % page)
                        request = urllib.request.urlopen(page, timeout=60)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        bytes = request.read()
                        if __is_image(bytes):
                            with open(out_dir + os.path.sep + page.split('/')[-1], "wb+") as file:
                                file.write(bytes)
                        else:
                            print("The resource %s is not an image" % page)
                    else:
                        print("The resource %s does not exist" % page)
                except Exception:
                    print("Error downloading %s" % page)


def __if_exists(url):
    exists = False
    try:
        conn = requests.head(url)
        exists = conn.status_code in (200, 301, 302, 304)
    except Exception:
        pass
    return exists

def __is_image(bytes):
    is_image = False
    MAGIC_NUMBERS = [[0xFF, 0xD8, 0xFF, 0xD8], [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01],
                     [0xFF, 0xD8, 0xFF, 0xEE], [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
                     [0x47, 0x49, 0x46, 0x38, 0x37, 0x61], [0x47, 0x49, 0x46, 0x38, 0x39, 0x61],
                     [0xFF, 0xD8, 0xFF, 0xE1, -1, -1, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00]]
    for numbers in MAGIC_NUMBERS:
        for i in range(len(numbers)):
            if numbers[i] < 0:
                continue
            if numbers[i] != bytes[i]:
                break
            elif i == len(numbers) - 1:
                return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", help="Dataset output directory")
    parser.add_argument("-f", "--file", help="File with dataset links")
    parser.add_argument("-l", "--links", nargs="*", help="Links to datasets")

    args = parser.parse_args()
    if args.file is not None:
        get_dataset_from_file(args.file, args.out)
    if args.links is not None:
        get_dataset_from_links(args.links, args.out)