---
date: 2021-10-12 14:05
title: "Google Data Analytics Professional Certificate - Course 1 : Foundations: Data, Data, Everywhere - week 4"
categories: coursera google GoogleDataAnalytics
tags: coursera google GoogleDataAnalytics
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Week 4 - SQL
What is a query?

A query is a request for data or information from a database. When you query databases, you use SQL to communicate your question or request. You and the database can always exchange information as long as you speak the same language.

Every programming language, including SQL, follows a unique set of guidelines known as syntax. Syntax is the predetermined structure of a language that includes all required words, symbols, and punctuation, as well as their proper placement.


## Capitalization, indentation, and semicolons
You can write your SQL queries in all lowercase and don’t have to worry about extra spaces between words. However, using capitalization and indentation can help you read the information more easily.

```SQL
SELECT
    field1
FROM
    table
WHERE
    field1 = condition;
```

Notice that the SQL statement shown above has a semicolon at the end. The semicolon is a statement terminator and is part of the American National Standards Institute (ANSI) SQL-92 standard, which is a recommended common syntax for adoption by all SQL databases. However, not all SQL databases have adopted or enforce the semicolon, so it’s possible you may come across some SQL statements that aren’t terminated with a semicolon. If a statement works without a semicolon, it’s fine.

## WHERE conditions
the WHERE clause narrows your query so that the database returns only the data with an exact value match or the data that matches a certain condition that you want to satisfy. 

For example, if you are looking for a specific customer with the last name Chavez, the WHERE clause would be: 

```SQL
WHERE field1 = 'Chavez'
```
However, if you are looking for all customers with a last name that begins with the letters `“Ch,"` the WHERE clause would be:

```SQL
WHERE field1 LIKE 'Ch%'
```

You can conclude that the LIKE clause is very powerful because it allows you to tell the database to look for a certain pattern! The percent sign (%) is used as a wildcard to match one or more characters. In the example above, both Chavez and Chen would be returned. Note that in some databases an asterisk (*) is used as the wildcard instead of a percent sign (%).

## Comments
Comments are text placed between certain characters, /* and */, or after two dashes (--) as shown below. 

```SQL
SELECT
    field1 /* this is the last name column */
FROM
    table -- this is the customer data table
WHERE
    field1 LIKE 'Ch%'
```

## Aliases
You can also make it easier on yourself by assigning a new name or alias to the column or table names to make them easier to work with.

## More
you want to make sure your results give you only the full time employees with salaries that are {% raw %}$30,000{% endraw %} or less. In other words, you want to exclude interns with the INT job code who also earn less than $30,000. The AND clause enables you to test for both conditions. 

You create a SQL query similar to below, where <> means "does not equal":

``` SQL
SELECT
    *
FROM
    Employee
WHERE
    jobCode <> 'INT'
    AND salary <= 30000;
```
```py
import dd
```

<!-- ![data-analysis-process-6-phase](\assets\images\data-analysis-process-6-phase.png){: .align-center .img-80}   -->

# Week 4 - Data Visualization
## Step 1: Explore the data for patterns

While reviewing the data you notice a pattern among those who visit the company’s website most frequently: geography and larger amounts spent on purchases. With further analysis, this information might explain why sales are so strong right now in the northeast—and help your company find ways to make them even stronger through the new website. 

## Step 2: Plan your visuals
Next it is time to refine the data and present the results of your analysis. Right now, you have a lot of data spread across several different tables, which isn’t an ideal way to share your results with management and the marketing team. You will want to create a data visualization that explains your findings quickly and effectively to your target audience. Since you know your audience is sales oriented, you already know that the data visualization you use should:

- Show sales numbers over time
- Connect sales to location
- Show the relationship between sales and website use
- Show which customers fuel growth

## Step 3: Create your visuals
Now that you have decided what kind of information and insights you want to display, it is time to start creating the actual visualizations. Keep in mind that creating the right visualization for a presentation or to share with stakeholders is a process. It involves trying different visualization formats and making adjustments until you get what you are looking for. In this case, a mix of different visuals will best communicate your findings and turn your analysis into the most compelling story for stakeholders. So, you can use the built-in chart capabilities in your spreadsheets to organize the data and create your visuals.


# Appendix
## Glossary: Terms and definitions


## Reference
- W3Schools SQL Tutorial: If you would like to explore a detailed tutorial of SQL, this is the perfect place to start. This tutorial includes interactive examples you can edit, test, and recreate. Use it as a reference or complete the whole tutorial to practice using SQL. Click the green Start learning SQL now button or the Next button to begin the tutorial.
    - <https://www.w3schools.com/sql/default.asp>
- SQL Cheat Sheet: For more advanced learners, go through this article for standard SQL syntax used in PostgreSQL. By the time you are finished, you will know a lot more about SQL and will be prepared to use it for business analysis and other tasks.
    - <https://towardsdatascience.com/sql-cheat-sheet-776f8e3189fa>
