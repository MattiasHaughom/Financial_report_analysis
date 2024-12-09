# Financial Report Processing Script
This script processes financial PDF reports by extracting relevant pages based on specific keywords and deleting the original files. It is designed to prepare data for AI analysis, which will summarize and store the information for future use.

## Features
- PDF Processing: Extracts pages from PDF reports that contain keywords such as "highlight", "summary", "key figures", "profit", and "profit margin".
- File Management: Deletes the original PDF files after extracting the relevant pages to save storage space.
- Logging: Logs the processing steps and any errors encountered during execution.

## Requirements
- Python 3.x
- Libraries: requests, pdfplumber, PyPDF2

## Install the required libraries using pdm:
```
pdm init
```

## Usage
1. Setup: Place the PDF reports you want to process in the downloads directory.
2. Run the Script: Execute the script to process the reports.
```
python3 report_scraping.py
```
3. Output: The script will create new PDF files containing only the relevant pages in the same directory, prefixed with extracted_.

## Configuration
- Keywords: Modify the list of keywords in the process_reports function to adjust which pages are extracted.
- Directory: Change the directory variable in the process_reports function to specify a different directory for input and output files.

## Future Enhancements
- AI Analysis Integration
- Data Extraction: Implement AI models to analyze the extracted data and generate summaries.
- Storage: Save the AI-generated summaries in a database or file system for future retrieval and analysis.
- User Interface: Develop a user interface to visualize the analysis results and allow users to interact with the data.
- Additional Features
- Batch Processing: Enhance the script to process large volumes of reports more efficiently.
- Parallel Processing: Utilize parallel processing to speed up the extraction and analysis of reports.
- Advanced Logging: Implement more detailed logging to track the performance and outcomes of the AI analysis.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.