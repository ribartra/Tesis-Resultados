-- Tabla de hilos de chat
CREATE TABLE IF NOT EXISTS chat_threads (
    id_thread   SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabla de mensajes
CREATE TABLE IF NOT EXISTS chat_messages (
    id          SERIAL PRIMARY KEY,
    thread_id   INTEGER NOT NULL REFERENCES chat_threads(id_thread) ON DELETE CASCADE,
    role        VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    message     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tabla de ratings de interacción
CREATE TABLE IF NOT EXISTS interaction_ratings (
    id_rating        SERIAL PRIMARY KEY,
    user_msg_id      INTEGER NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    assistant_msg_id INTEGER NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    score            INTEGER NOT NULL CHECK (score BETWEEN 1 AND 10),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Índices recomendados para rendimiento
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created
    ON chat_messages(thread_id, created_at);

CREATE INDEX IF NOT EXISTS idx_interaction_ratings_user_msg
    ON interaction_ratings(user_msg_id);

CREATE INDEX IF NOT EXISTS idx_interaction_ratings_assistant_msg
    ON interaction_ratings(assistant_msg_id);
