DROP TABLE IF EXISTS events CASCADE;

CREATE TABLE events (
    id              BIGSERIAL PRIMARY KEY,
    event_time      TIMESTAMP WITH TIME ZONE,
    event_type      VARCHAR(20),
    product_id      BIGINT,
    category_id     BIGINT,
    category_code   VARCHAR(255),
    brand           VARCHAR(100),
    price           NUMERIC(10, 2),
    user_id         BIGINT,
    user_session    VARCHAR(50)
);

CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_user_id    ON events(user_id);