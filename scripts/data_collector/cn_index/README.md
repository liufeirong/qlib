# CSI300/CSI100 History Companies Collection

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# parse instruments, using in qlib/instruments.
python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/qlib_cn_1d --method parse_instruments

# parse new companies
python collector.py --index_name CSI300 --qlib_dir ~/.qlib/qlib_data/qlib_cn_1d --method save_new_companies


# index_name support: CSI300, CSI100
# help
python collector.py --help
```

