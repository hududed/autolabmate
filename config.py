import os
from enum import Enum

from pydantic_settings import BaseSettings


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


environment = os.environ.get("APP_ENV", Environment.DEVELOPMENT.value)

env_configs = {
    "development": {
        "domain": "autolabmate.local",
        "backend_url": "http://api.autolabmate.local:8000",
        "frontend_url": "http://app.autolabmate.local:3000",
    },
    "production": {
        "domain": "autolabmate.com",
        "backend_url": "https://api.autolabmate.com",
        "frontend_url": "https://app.autolabmate.com",
    },
}

domain = env_configs[environment]["domain"]
backend_url = env_configs[environment]["backend_url"]
frontend_url = env_configs[environment]["frontend_url"]


class AuthProviders(Enum):
    GITHUB = "github"
    PASSWORDLESS = "passwordless"
    WITH_EMAIL_PASSWORD = "with_email_password"


auth_providers = {
    AuthProviders.GITHUB.value,
    AuthProviders.PASSWORDLESS.value,
    AuthProviders.WITH_EMAIL_PASSWORD.value,
}

features = {
    "upload_file": environment == Environment.PRODUCTION.value,
}


class Settings(BaseSettings):
    # General
    ENVIRONMENT: str = environment
    DOMAIN: str = domain
    BACKEND_URL: str = backend_url
    FRONTEND_URL: str = frontend_url
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    # Auth
    AUTH_PROVIDERS: set = auth_providers
    AUTH_TOKEN_COOKIE_NAME: str = "auth_token"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_SECRET_KEY: str = "J3gWSn3zTL/cOryi5BjzNaNUxm2J9klvgKc9MJVv86w="
    ACCESS_TOKEN_EXPIRE_DAYS: int = 30
    INVITE_TOKEN_SECRET_KEY: str = "IRCBwaEXG4zsyPdJw7YHB6dV5BE5HdVmfH5eJO3cdDk="
    INVITE_TOKEN_EXPIRE_MINUTES: int = 5
    # Database
    SUPABASE_URL: str = "supabase_url"
    SUPABASE_KEY: str = "supabase_key"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    DB_HOST: str = "db"
    DB_PORT: int = 5432
    DB_NAME: str = "postgres"
    # Google Cloud Storage
    # GOOGLE_STORAGE_BUCKET: str = None
    # GOOGLE_CLOUD_PROJECT: str = None
    # MAX_FILE_SIZE: int = 4000000
    # # Github OAuth
    # GITHUB_CLIENT_ID: str = None
    # GITHUB_CLIENT_SECRET: str = None

    FEATURES: dict[str, bool] = features

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
