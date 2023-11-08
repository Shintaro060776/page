import os
import csv

folder_path = '/page/Docker/rendered_256x256/cat/'

output_file = 'foldername1.csv'

with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename'])

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            writer.writerow([filename])

print(f'File names have written to {output_file}')
