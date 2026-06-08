import { Hono } from 'hono';
import { saveInquiry } from '../db/queries.js';
import { sendInquiryEmail } from '../lib/email.js';

export const inquiries = new Hono();

function str(v: unknown): string {
  return typeof v === 'string' ? v.trim() : '';
}

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

inquiries.post('/', async (c) => {
  let body: Record<string, unknown>;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: 'Invalid JSON' }, 400);
  }

  // Honeypot: a hidden field humans never see. If it's filled, it's a bot —
  // silently accept (200) so the bot thinks it succeeded, but drop the record.
  if (str(body._honey)) return c.json({ ok: true });

  const name = str(body.name);
  const email = str(body.email);
  const message = str(body.message);
  const phone = str(body.phone);
  const company = str(body.company);
  const machineId = str(body.machineId);
  const locale = str(body.locale);

  if (!name || !email || !message) {
    return c.json({ error: 'Missing required fields' }, 400);
  }
  if (!EMAIL_RE.test(email) || email.length > 200) {
    return c.json({ error: 'Invalid email' }, 400);
  }
  if (name.length > 200 || company.length > 200 || message.length > 5000) {
    return c.json({ error: 'Field too long' }, 400);
  }

  const input = {
    name,
    email,
    phone: phone || undefined,
    company: company || undefined,
    machineId: machineId || undefined,
    message,
    locale: locale || undefined,
  };

  // Email is the primary delivery channel — the lead must reach the inbox.
  const emailed = await sendInquiryEmail(input);
  // DB persistence is best-effort: never lose a lead over a storage hiccup.
  await saveInquiry(input).catch((err) =>
    console.warn('[inquiry] DB store failed (non-fatal):', err),
  );

  if (!emailed) {
    return c.json({ error: 'Could not deliver inquiry. Please email us directly.' }, 502);
  }
  return c.json({ ok: true }, 201);
});
