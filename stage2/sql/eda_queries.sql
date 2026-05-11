INSERT OVERWRITE DIRECTORY 'project/output/q1'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT event_type, COUNT(*) as cnt, COUNT(DISTINCT user_id) as unique_users
FROM events_partitioned
GROUP BY event_type;

INSERT OVERWRITE DIRECTORY 'project/output/q2'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT product_id, COUNT(*) as views
FROM events_partitioned
WHERE event_type = 'view'
GROUP BY product_id
ORDER BY views DESC
LIMIT 10;

INSERT OVERWRITE DIRECTORY 'project/output/q3'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT product_id, COUNT(*) as purchases
FROM events_partitioned
WHERE event_type = 'purchase'
GROUP BY product_id
ORDER BY purchases DESC
LIMIT 10;

INSERT OVERWRITE DIRECTORY 'project/output/q4'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT 
    v.product_id,
    v.views,
    COALESCE(p.purchases, 0) as purchases,
    ROUND(COALESCE(p.purchases, 0) * 100.0 / v.views, 2) as conversion_rate
FROM 
    (SELECT product_id, COUNT(*) as views FROM events_partitioned WHERE event_type = 'view' GROUP BY product_id) v
LEFT JOIN
    (SELECT product_id, COUNT(*) as purchases FROM events_partitioned WHERE event_type = 'purchase' GROUP BY product_id) p
ON v.product_id = p.product_id
WHERE v.views > 100
ORDER BY conversion_rate DESC
LIMIT 20;

INSERT OVERWRITE DIRECTORY 'project/output/q5'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT 
    user_id,
    COUNT(*) as total_events,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases,
    SUM(price) as total_spent
FROM events_partitioned
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 50;
