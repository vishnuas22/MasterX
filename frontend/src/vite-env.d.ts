/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_BACKEND_URL: string;
  readonly VITE_APP_NAME: string;
  readonly VITE_APP_VERSION: string;
  readonly VITE_WS_URL: string;
  readonly VITE_ENABLE_VOICE: string;
  readonly VITE_ENABLE_ANALYTICS: string;
  readonly VITE_ENABLE_GAMIFICATION: string;
  readonly VITE_GA_TRACKING_ID: string;
  readonly VITE_SENTRY_DSN: string;
  readonly VITE_ENVIRONMENT: string;
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly MODE: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
