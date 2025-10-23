import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import axios from 'axios';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

const uploadDir = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, uniqueSuffix + '-' + file.originalname.replace(/\s+/g, '_'));
  },
});

const upload = multer({ storage });

app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// Fields: description (string), summary (string optional)
// Files: multiple under fields: before, during, after
app.post(
  '/api/relevancy',
  upload.fields([
    { name: 'before', maxCount: 10 },
    { name: 'during', maxCount: 10 },
    { name: 'after', maxCount: 10 },
  ]),
  async (req, res) => {
    try {
      const description = req.body.description || '';
      const summary = req.body.summary || '';

      if (!description) {
        return res.status(400).json({ error: 'description is required' });
      }

      const toBase64 = (filePath) => {
        const data = fs.readFileSync(filePath);
        return data.toString('base64');
      };

      const collectGroup = (groupName) => {
        const files = (req.files && req.files[groupName]) || [];
        return files.map((f) => ({ filename: f.originalname, b64: toBase64(f.path) }));
      };

      const groups = {
        before: collectGroup('before'),
        during: collectGroup('during'),
        after: collectGroup('after'),
      };

      const pyServiceUrl = process.env.PY_SERVICE_URL || 'http://127.0.0.1:8000';

      const payload = {
        description,
        summary,
        images: {
          before: groups.before.map((x) => x.b64),
          during: groups.during.map((x) => x.b64),
          after: groups.after.map((x) => x.b64),
        },
      };

      const response = await axios.post(`${pyServiceUrl}/score`, payload, {
        timeout: 1000 * 60 * 5,
        headers: { 'Content-Type': 'application/json' },
      });

      const result = response.data;

      // Attach original filenames back to results for client convenience
      const attachFilenames = (items, filenames) => {
        return items.map((it, idx) => ({ ...it, filename: filenames[idx] || null }));
      };

      result.groups = result.groups || {};
      result.groups.before = attachFilenames(
        result.groups.before || [],
        groups.before.map((x) => x.filename)
      );
      result.groups.during = attachFilenames(
        result.groups.during || [],
        groups.during.map((x) => x.filename)
      );
      result.groups.after = attachFilenames(
        result.groups.after || [],
        groups.after.map((x) => x.filename)
      );

      res.json(result);
    } catch (err) {
      // Avoid leaking secrets
      const message = err?.response?.data || err?.message || 'Unknown error';
      res.status(500).json({ error: 'Failed to score relevancy', details: message });
    } finally {
      // Clean up uploaded files
      try {
        const allFiles = [
          ...(((req.files && req.files.before) || []).map((f) => f.path)),
          ...(((req.files && req.files.during) || []).map((f) => f.path)),
          ...(((req.files && req.files.after) || []).map((f) => f.path)),
        ];
        for (const fp of allFiles) {
          if (fp && fs.existsSync(fp)) {
            fs.unlinkSync(fp);
          }
        }
      } catch (_) {}
    }
  }
);

app.listen(port, () => {
  // Do not log secrets
  console.log(`Node API listening on port ${port}`);
});


