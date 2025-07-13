import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'MasterX - AI-Powered Learning Platform',
  description: 'Revolutionary quantum intelligence learning platform with advanced AI mentorship',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative">
          {/* Quantum Particles Background */}
          <div className="quantum-bg">
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            <div className="quantum-particle"></div>
            
            <div className="quantum-orb"></div>
            <div className="quantum-orb"></div>
            <div className="quantum-orb"></div>
          </div>
          
          {/* Main Content */}
          <div className="relative z-10">
            {children}
          </div>
        </div>
      </body>
    </html>
  )
}