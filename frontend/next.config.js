/** @type {import('next').NextConfig} */
const path = require('path')

const nextConfig = {
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: false
  },
  images: {
    domains: ['avatars.githubusercontent.com'],
  },
  experimental: {
    appDir: true
  },
  // Import path aliases
  webpack: (config) => {
    config.resolve.alias['@'] = path.resolve(__dirname, 'src')
    return config
  },
  // Ensure proper routing
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.REACT_APP_BACKEND_URL + '/api/:path*'
      }
    ]
  }
}

module.exports = nextConfig