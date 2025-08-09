import './globals.css'
import { Inter } from 'next/font/google'
import { AuthProvider } from '@/contexts/AuthContext'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap'
})

export const metadata = {
  title: 'MasterX - Quantum Intelligence Platform',
  description: 'Revolutionary AI-powered learning platform with quantum intelligence and ultra-premium interface',
  keywords: 'AI, machine learning, quantum intelligence, education, premium interface',
  authors: [{ name: 'MasterX Team' }],
  colorScheme: 'dark',
  themeColor: '#A855F7',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
      </head>
      <body className={`${inter.variable} font-inter antialiased min-h-screen bg-quantum-dark text-plasma-white`}>
        <AuthProvider>
          <div className="relative">
            {children}
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}