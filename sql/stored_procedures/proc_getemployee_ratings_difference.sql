DELIMITER $$

CREATE PROCEDURE GetEmployeeRatingsDifference()
BEGIN
    SELECT
        e.employee_id,
        e.name,
        er.self_rating_l3q,
        er.manager_rating_l3q,
        CASE 
            WHEN er.self_rating_l3q > er.manager_rating_l3q THEN 'Yes'
            ELSE 'No'
        END AS self_greater_than_manager
    FROM employees e
    JOIN employee_ratings er ON e.employee_id = er.employee_id;
END$$

DELIMITER ;
