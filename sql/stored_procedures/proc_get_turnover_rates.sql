DELIMITER $$

CREATE PROCEDURE GetTurnoverRates()
BEGIN
    SELECT 
        r.role_name,
        l.location_name,
        COUNT(*) AS total_employees,
        SUM(CASE WHEN e.active = FALSE THEN 1 ELSE 0 END) AS inactive_employees,
        ROUND(100.0 * SUM(CASE WHEN e.active = FALSE THEN 1 ELSE 0 END) / COUNT(*), 2) AS turnover_percentage
    FROM employees e
    JOIN roles r ON e.role_id = r.role_id
    JOIN locations l ON e.location_id = l.location_id
    GROUP BY r.role_name, l.location_name
    ORDER BY turnover_percentage DESC;
END$$

DELIMITER ;
