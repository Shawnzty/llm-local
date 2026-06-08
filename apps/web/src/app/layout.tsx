import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Tadzuna',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
