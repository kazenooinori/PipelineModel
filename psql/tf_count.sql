select tf, count(*) from
(
select t.dt, t.params, x2.timediff as tf from 
  (
   SELECT dt, string_agg(concat(tagname, ';', value), ';') as params
   FROM process_var
   GROUP BY dt
  ) t
  left outer join 
(
select dt,   
extract ( seconds from age(dt, lag(dt) over (order by dt))) as timediff
 from
  (
    select s.*, 
      (case when lag(value) over (order by dt) = value
       then 0 else 1
       end) as startflag
    from weight_var s order by dt 
  ) x1 where startflag = 1
) x2
on t.dt = x2.dt
) oh group by tf
order by tf;
