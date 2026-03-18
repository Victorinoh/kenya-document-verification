import shutil, os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = f'models/backups/{timestamp}'
os.makedirs(backup_dir, exist_ok=True)

for f in ['models/saved_models/document_classifier.h5',
          'models/saved_models/authenticity_detector.h5']:
    if os.path.exists(f):
        shutil.copy2(f, backup_dir)
        print(f'Backed up: {f}')
    else:
        print(f'Not found: {f}')

print(f'Backup saved to: {backup_dir}')
