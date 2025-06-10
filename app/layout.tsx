import "../styles/globals.css";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Alpha Mentor",
  description: "A world-class, futuristic AI Mentor",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-dark-900 min-h-screen flex flex-col antialiased`}>
        {children}
      </body>
    </html>
  );
}