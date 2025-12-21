
import gzip
from bs4 import BeautifulSoup
import os

target_file = 'data/html/race/202206020703.html.gz'

def try_decode(content, encoding):
    print(f"\n--- Trying {encoding} ---")
    try:
        decoded = content.decode(encoding)
        print("Success!")
        print(decoded[:200])
        return decoded
    except Exception as e:
        print(f"Failed: {e}")
        return None

if os.path.exists(target_file):
    with gzip.open(target_file, 'rb') as f:
        raw_content = f.read()
    
    print(f"First 20 bytes (hex): {raw_content[:20].hex()}")
    
    # Try common encodings
    # try_decode(raw_content, 'euc-jp')
    decoded_utf8 = try_decode(raw_content, 'utf-8')
    if decoded_utf8:
        soup = BeautifulSoup(decoded_utf8, 'lxml')
        title = soup.find('title')
        print(f"Title (UTF-8): {title.text if title else 'Not Found'}")
    
    # try_decode(raw_content, 'cp932')
    # try_decode(raw_content, 'shift_jis')
    
else:
    print(f"File not found: {target_file}")
