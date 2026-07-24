import os
import sys

def main():
    print("==================================================")
    print("Exoplanet Bulk Data Downloader (AWS Cloud Ingestion)")
    print("==================================================")
    
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        print("[!] Missing AWS SDK. Installing 'boto3' is required to fetch TESS/Kepler public data in bulk.")
        print("    Please run: pip install boto3")
        print("\nAlternatively, you can run the AWS CLI from your command prompt:")
        print("    aws s3 sync --no-sign-request s3://stpubdata/tess/public/lightcurves/data/tess/ ./data/raw_fits/")
        return

    # User Configuration
    max_files = 30000  # 30,000 light curve FITS files ~ 45 GB (each FITS is ~1.5 MB)
    target_mission = "tess"  # 'tess' or 'kepler'
    output_dir = "data/raw_fits"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure boto3 to access the public AWS bucket anonymously
    print(f"[*] Accessing anonymous public S3 bucket: s3://stpubdata/{target_mission}/...")
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    bucket_name = "stpubdata"
    
    # Define search prefix for light curve files
    if target_mission == "tess":
        # TESS public light curves are located in tess/public/tid/
        prefix = "tess/public/tid/"
    else:
        # Kepler public light curves
        prefix = "kepler/lightcurves/"
        
    print(f"[*] Scanning bucket directories for FITS files starting with '{prefix}'...")
    
    # Paginate through the bucket objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    count = 0
    fits_keys = []
    
    try:
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                # We only want standard light curve fits files (avoiding target pixel files etc)
                if key.endswith('_lc.fits') or (target_mission == 'kepler' and key.endswith('.fits') and 'lightcurves' in key):
                    fits_keys.append(key)
                    if len(fits_keys) >= max_files:
                        break
            if len(fits_keys) >= max_files:
                break
    except Exception as e:
        print(f"[!] Error scanning bucket: {e}")
        return
        
    total_found = len(fits_keys)
    print(f"[+] Found {total_found} available light curve files to download.")
    
    if total_found == 0:
        print("[!] No matching files found. Check your mission configuration prefix.")
        return
        
    confirm = input(f"[?] Proceed to download {total_found} files (~40-50 GB)? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("[i] Download aborted by user.")
        return
        
    print(f"[*] Starting ingestion. Writing files directly to: {output_dir}")
    
    for idx, key in enumerate(fits_keys):
        file_name = os.path.basename(key)
        local_path = os.path.join(output_dir, file_name)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            count += 1
            continue
            
        try:
            # Download file from S3
            s3.download_file(bucket_name, key, local_path)
            count += 1
            
            # Print progress every 50 files
            if count % 50 == 0 or count == total_found:
                progress = (count / total_found) * 100
                est_gb = (count * 1.5) / 1024
                print(f"    Downloaded: {count}/{total_found} | Progress: {progress:.2f}% | Est. Size: {est_gb:.2f} GB")
                
        except KeyboardInterrupt:
            print("\n[!] Download interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[!] Error downloading {file_name}: {e}")
            
    print(f"\n[+] Bulk Ingestion Complete! Successful downloads: {count}/{total_found}")
    print(f"    Data is saved in: {output_dir}")

if __name__ == "__main__":
    main()
