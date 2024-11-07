from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    # SERVER
    SERVER_HOST: str
    SERVER_PORT: int

    # ENVIRONMENT
    OUTPUT_DIR: str
    HUGGINGFACEHUB_API_TOKEN: str
    DEVICE_NUM: str


# 환경 변수 로드 및 검증
settings = Settings()
