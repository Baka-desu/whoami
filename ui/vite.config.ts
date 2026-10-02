import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The console talks only to Dev 5's operator gateway (ugv_nav/ugv_api). In dev the gateway is proxied so the
// browser stays same-origin; override the target with UGV_API_URL (e.g. a robot on the LAN).
const gateway = process.env.UGV_API_URL ?? 'http://127.0.0.1:8080'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': { target: gateway, changeOrigin: true } } },
  preview: { proxy: { '/api': { target: gateway, changeOrigin: true } } },
})
