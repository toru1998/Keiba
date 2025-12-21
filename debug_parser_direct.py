
import logging
from src.preprocessing.html_parser import NetkeibaHtmlParser
import json

logging.basicConfig(level=logging.INFO)

def test_parser():
    parser = NetkeibaHtmlParser()
    target_file = 'data/html/race/202206020703.html.gz'
    
    print(f"Testing parse on {target_file}")
    result = parser.parse_file(target_file)
    
    if result:
        print("\n--- Race Info ---")
        print(json.dumps(result['race_info'], ensure_ascii=False, indent=2))
        
        print("\n--- First 5 Results ---")
        if result['results'] is not None:
            print(result['results'].head().to_string())
    else:
        print("Parse failed.")

if __name__ == "__main__":
    test_parser()
