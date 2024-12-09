# Financial Report Processing Script
This app automates the retrieval, processing, and preparation of financial reports for further analysis.

## report_scraping.py
This script is responsible for fetching financial reports from an API, downloading relevant attachments, and managing the storage of these files.

### Features
- Fetch Message IDs: Retrieves message IDs from the API based on specific categories and market criteria.
- Download Attachments: Downloads attachments from the messages that are identified as financial reports.
- File Management: Deletes files older than a specified number of days to manage storage space.
- Logging: Logs the operations and any errors encountered during execution.

## process_reports.py
This script processes the downloaded PDF reports by extracting relevant pages based on specific keywords and deleting the original files.

### Features
- Extract Relevant Pages: Identifies and extracts pages from PDF reports that contain keywords such as "highlight", "summary", "key figures", "profit", and "profit margin".
- Save Extracted Pages: Saves the extracted pages into new PDF files.
- Delete Original Files: Deletes the original PDF files after processing to save storage space.


## Requirements
- Python 3.x
- Libraries: requests, pdfplumber, PyPDF2

## Install the required libraries using pdm:
```
pdm init
```


## Future Enhancements
### AI Analysis Integration
- Data Extraction: Implement AI models to analyze the extracted data and generate summaries.
- Storage: Save the AI-generated summaries in a database or file system for future retrieval and analysis.
- User Interface: Develop a user interface to visualize the analysis results and allow users to interact with the data.
### Additional Features
- Batch Processing: Enhance the script to process large volumes of reports more efficiently.
- Parallel Processing: Utilize parallel processing to speed up the extraction and analysis of reports.
- Advanced Logging: Implement more detailed logging to track the performance and outcomes of the AI analysis.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.