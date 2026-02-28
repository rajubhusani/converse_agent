-- ConverseAgent — MySQL initialization
-- This runs automatically on first docker-compose start.

-- Ensure utf8mb4 for full Unicode support (Hindi, emoji, etc.)
ALTER DATABASE converse_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Grant all privileges to the app user
GRANT ALL PRIVILEGES ON converse_agent.* TO 'converse'@'%';
FLUSH PRIVILEGES;
