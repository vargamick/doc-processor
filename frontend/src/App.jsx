import React, { useState } from 'react'
import './App.css'

function App() {
    const [file, setFile] = useState(null)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0]
        setFile(selectedFile)
        setError(null)
        setResult(null)
    }

    const handleSubmit = async (event) => {
        event.preventDefault()
        if (!file) {
            setError('Please select a file')
            return
        }

        try {
            setLoading(true)
            setError(null)

            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch('http://localhost:8000/documents/process', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`)
            }

            const data = await response.json()
            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="container">
            <h1>Doc Processor</h1>
            
            <form onSubmit={handleSubmit} className="form">
                <div className="file-input-container">
                    <input
                        type="file"
                        accept=".pdf,.docx"
                        onChange={handleFileChange}
                        className="file-input"
                    />
                    <button 
                        type="submit" 
                        disabled={!file || loading}
                        className="submit-button"
                    >
                        {loading ? 'Processing...' : 'Process Document'}
                    </button>
                </div>
            </form>

            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {result && (
                <div className="result-container">
                    <h3>Processed Text:</h3>
                    <pre className="result-text">
                        {result.text}
                    </pre>
                </div>
            )}
        </div>
    )
}

export default App
