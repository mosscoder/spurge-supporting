from datasets import load_dataset
import os

local_dir = 'spurge_data'

for dataset in ['crop', 'context']:
    for split in ['train', 'test']:
        formatted_dir = os.path.join(local_dir, dataset, split)
        os.makedirs(formatted_dir, exist_ok=True)
        
        data = load_dataset('mpg-ranch/leafy_spurge', dataset, split=split)
        # Convert dataset to dataframe, excluding image column
        meta_df = data.to_pandas()
        meta_df = meta_df.drop('image', axis=1)
        
        # Save metadata to CSV file
        metadata_path = os.path.join(formatted_dir, 'metadata.csv')
        meta_df.to_csv(metadata_path, index=False)
        
        for i in range(len(data)):
            observation = data[i]
            # Extract the image and metadata
            image = observation['image']
            idx = observation['idx']
            label = 'present' if observation['label'] == 1 else 'absent'

            fname = os.path.join(label, f'{idx}.png')
            image_path = os.path.join(formatted_dir, fname)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            image.save(image_path)