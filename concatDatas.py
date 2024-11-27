import pandas as pd

# Read the first Excel sheet
sheet1 = pd.read_excel("Healthy_Biomarker_Features.xlsx")

# Read the second Excel sheet
sheet2 = pd.read_excel('NSCLC_Biomarker_Features.xlsx')

# Concatenate the two sheets vertically (axis=0)
combined = pd.concat([sheet1, sheet2], axis=0, ignore_index=True)

# Save the result to a new Excel file (optional)
combined.to_excel('combined_file.xlsx', index=False)

print("Sheets concatenated successfully!")
