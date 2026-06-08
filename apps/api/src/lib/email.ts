import { MACHINES } from '@tadzuna/shared';

export interface InquiryInput {
  name: string;
  email: string;
  phone?: string;
  company?: string;
  machineId?: string;
  message: string;
  locale?: string;
}

const RESEND_ENDPOINT = 'https://api.resend.com/emails';

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function machineLabel(machineId?: string): string {
  if (!machineId) return '—';
  const m = MACHINES.find((x) => x.id === machineId);
  return m ? `${m.name} (${machineId})` : machineId;
}

/**
 * Send an inquiry notification to the sales inbox via the Resend REST API.
 *
 * Returns true on success. Requires RESEND_API_KEY — if it's unset (e.g. local
 * dev), this logs a warning and returns false so the caller can decide what to do.
 * Uses global fetch (Node 20+), so no extra dependency is needed.
 */
export async function sendInquiryEmail(input: InquiryInput): Promise<boolean> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    console.warn('[inquiry] RESEND_API_KEY not set — cannot send notification email.');
    return false;
  }

  const from = process.env.INQUIRY_FROM ?? 'Tadzuna <noreply@tadzuna.com>';
  const to = process.env.INQUIRY_TO ?? 'info@tadzuna.com';

  const rows: [string, string][] = [
    ['Name', input.name],
    ['Email', input.email],
    ['Phone', input.phone || '—'],
    ['Company', input.company || '—'],
    ['Machine', machineLabel(input.machineId)],
    ['Locale', input.locale || '—'],
  ];

  const text =
    `New inquiry from ${input.name}\n\n` +
    rows.map(([k, v]) => `${k}: ${v}`).join('\n') +
    `\n\nMessage:\n${input.message}\n`;

  const html =
    `<h2 style="font-family:system-ui,sans-serif">New inquiry</h2>` +
    `<table cellpadding="6" style="border-collapse:collapse;font-family:system-ui,sans-serif;font-size:14px">` +
    rows
      .map(
        ([k, v]) =>
          `<tr><td style="color:#666;vertical-align:top">${esc(k)}</td><td><strong>${esc(v)}</strong></td></tr>`,
      )
      .join('') +
    `</table>` +
    `<p style="font-family:system-ui,sans-serif;font-size:14px"><strong>Message</strong><br>${esc(
      input.message,
    ).replace(/\n/g, '<br>')}</p>`;

  try {
    const res = await fetch(RESEND_ENDPOINT, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        from,
        to,
        reply_to: input.email,
        subject: `[Tadzuna] New inquiry — ${input.name}`,
        text,
        html,
      }),
    });
    if (!res.ok) {
      console.error('[inquiry] Resend error', res.status, await res.text().catch(() => ''));
      return false;
    }
    return true;
  } catch (err) {
    console.error('[inquiry] Resend request failed:', err);
    return false;
  }
}
