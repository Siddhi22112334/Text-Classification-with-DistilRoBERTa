import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def generate_overlapping_sections(elements, window_size):
    overlapping_sections = []
    for i in range(len(elements) - window_size + 1):
        section = ' '.join(elements[i:i + window_size])
        overlapping_sections.append(section)
    return overlapping_sections

def divide_and_augment(text):
    # Split the text into paragraphs
    paragraphs = text.split('\n') 

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Generate overlapping three-sentence sections
    three_line_sections = generate_overlapping_sections(sentences, 3)

    # Generate overlapping two-paragraph sections
    two_paragraph_sections = []
    paragraphs = text.split('\n\n')
    for i in range(len(paragraphs) - 1):
        combined_section = paragraphs[i] + '\n\n' + paragraphs[i + 1]
        two_paragraph_sections.append(combined_section)

    return three_line_sections, paragraphs, two_paragraph_sections

# Read the original CSV file
df = pd.read_csv('data/full.csv')

# Clean the data: Ensure all text entries are strings and handle NaN values
df['text'] = df['text'].astype(str).fillna('')

# Initialize lists to store the new datasets
three_sentence_data = []
single_paragraph_data = []
two_paragraph_data = []

# Process each text in the dataset
for _, row in df.iterrows():
    text = row['text']
    label = row['label']
    
    three_line_sections, single_paragraphs, two_paragraph_sections = divide_and_augment(text)
    
    for section in three_line_sections:
        three_sentence_data.append([section, label])
    
    for paragraph in single_paragraphs:
        single_paragraph_data.append([paragraph, label])
    
    for section in two_paragraph_sections:
        two_paragraph_data.append([section, label])

# Create DataFrames for each new dataset
three_sentence_df = pd.DataFrame(three_sentence_data, columns=['text', 'label'])
single_paragraph_df = pd.DataFrame(single_paragraph_data, columns=['text', 'label'])
two_paragraph_df = pd.DataFrame(two_paragraph_data, columns=['text', 'label'])

# Save the new DataFrames to CSV files
three_sentence_df.to_csv('data/three_sentence_dataset.csv', index=False)
single_paragraph_df.to_csv('data/single_paragraph_dataset.csv', index=False)
two_paragraph_df.to_csv('data/two_paragraph_dataset.csv', index=False)

print("CSV files for three sentences, one paragraph, and two paragraphs created successfully.")
