from pydantic import BaseModel


class Document(BaseModel):
    id: int
    title: str | None = None
    preview: str | None = None
    score: float | None = None

class DocumentDetail(Document):
    content: str


class PaginatedDocuments(BaseModel):
    current_page: int
    last_page: int
    per_page: int
    total: int
    data: list[Document]
