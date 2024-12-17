const fs = require('fs');
const path = require('path');
const csvParser = require('csv-parser');

// 新 JSON 資料
const newJsonData = [
  {"timestamp": "2024-06-17T12:34:56Z", "neutral": "0.0000", "happy": "0.0000", "sad": "0.0000", "surprised": "0.9998"}
];

// 原始 CSV 檔案路徑
const inputCSVPath = path.join(__dirname, 'input.csv');
const outputCSVPath = path.join(__dirname, 'combined_output.csv');

let existingData = [];

// 讀取原始 CSV 檔案
fs.createReadStream(inputCSVPath)
  .pipe(csvParser())
  .on('data', (row) => {
    existingData.push(row);
  })
  .on('end', () => {
    console.log("CSV 讀取完成");
    
    // 合併新資料
    const combinedData = [...existingData, ...newJsonData];

    // 寫入新的 CSV 檔案
    const headers = Object.keys(combinedData[0]);
    const csvContent = [
      headers.join(","), 
      ...combinedData.map(row => headers.map(header => row[header] || "").join(","))
    ].join("\n");

    fs.writeFileSync(outputCSVPath, csvContent);
    console.log("合併完成，輸出檔案路徑:", outputCSVPath);
  });
