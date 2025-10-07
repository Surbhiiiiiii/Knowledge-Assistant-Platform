import React, { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState("");
  const [answerText, setAnswerText] = useState("");
  const [sources, setSources] = useState([]);

  const uploadFile = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log("Upload success:", data);
      alert(`File uploaded successfully: ${data.filename}`);
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Upload failed. Check backend logs.");
    }
  };

  const ask = async (e) => {
    e.preventDefault();
    if (!query) return;

    try {
      const res = await fetch("http://127.0.0.1:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const j = await res.json();
      setAnswerText(j.answer || "");
      setSources(j.sources || [{ text: j.answer || JSON.stringify(j), filename: "Unknown" }]);
    } catch (err) {
      console.error("Chat request failed:", err);
      setAnswerText("Error getting response from backend.");
      setSources([]);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>AI Knowledge Assistant</h2>

      {/* Upload Section */}
      <form onSubmit={uploadFile}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit">Upload</button>
      </form>

      <hr />

      {/* Query Section */}
      <form onSubmit={ask}>
        <input
          type="text"
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ width: "70%", marginRight: "10px" }}
        />
        <button type="submit">Ask</button>
      </form>

      {/* Display Answer */}
      <div style={{ marginTop: "20px" }}>
        <strong>Assistant:</strong>
        <div style={{ whiteSpace: "pre-wrap" }}>{answerText || "No answer yet."}</div>

        <hr />

        <strong>Sources:</strong>
        {sources.map((s, idx) => (
          <div key={idx} style={{ marginTop: "10px" }}>
            <div style={{ fontStyle: "italic", color: "gray" }}>
              Source: {s.filename} {s.timestamp ? `(ts: ${s.timestamp})` : ""}
            </div>
            <div style={{ whiteSpace: "pre-wrap" }}>{s.preview || s.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
