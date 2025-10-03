export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).end();
  const user_id = req.query.user_id || 'anon';
  const issuer = process.env.BACKEND_URL || 'http://localhost:8000';
  const r = await fetch(`${issuer}/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id }),
  });
  const data = await r.json();
  res.status(r.status).json(data);
}
