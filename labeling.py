import os
import csv


def create_csv_labels(output_dir, csv_filename):

    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label'])

        for filename in os.listdir(output_dir):
            if filename.endswith('.png'):
                label = filename.split('_')[0]  # Extracting label from the filename
                writer.writerow([filename, label])



