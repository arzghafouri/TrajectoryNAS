import express from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());

// Initialize Google Gen AI
const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI(apiKey ? { apiKey } : {});

// API endpoints
app.post('/api/generate-code', async (req, res) => {
  try {
    const { prompt, framework, taskType } = req.body;
    
    const systemPrompt = `You are an expert Python AI and Machine Learning engineer. 
Write clean, robust, production-ready Python code using ${framework || 'PyTorch / Hugging Face'} for the task: ${taskType || 'General AI'}.
User request: ${prompt}

Provide your response in JSON format with two keys:
1. "files": an array of objects with "name" (e.g. "model.py", "train.py", "requirements.txt", "README.md") and "content" (the Python/text code).
2. "explanation": a concise explanation of the architecture and how to run it.`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: systemPrompt,
      config: {
        responseMimeType: 'application/json',
      }
    });

    const resultText = response.text;
    if (!resultText) {
      throw new Error('No response from Gemini API');
    }

    const parsed = JSON.parse(resultText);
    res.json(parsed);
  } catch (error: any) {
    console.error('Error generating AI code:', error);
    res.status(500).json({ error: error.message || 'Failed to generate code' });
  }
});

app.post('/api/debug-code', async (req, res) => {
  try {
    const { code, errorLog } = req.body;
    
    const prompt = `You are an expert Python debugger. Analyze this Python AI code and error log, explain the root cause, and provide the corrected code.

Python Code:
\`\`\`python
${code}
\`\`\`

Error Log / Issue:
\`\`\`
${errorLog}
\`\`\`

Return JSON with:
1. "analysis": explanation of why the bug occurred.
2. "fixedCode": the complete corrected Python code.`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
      }
    });

    res.json(JSON.parse(response.text || '{}'));
  } catch (error: any) {
    console.error('Error debugging code:', error);
    res.status(500).json({ error: error.message || 'Failed to debug code' });
  }
});

async function startServer() {
  const PORT = Number(process.env.PORT) || 3000;

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_, res) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
