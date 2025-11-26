# ==============================
# Prompt Library
# ==============================

# Prompt zur Keyword-Extraktion
extraction_prompt_template = """
You are a keyword extractor.
Extract only the essential product numbers, job IDs, or unique terms from this text.
For each extracted keyword, assign its type. 

Make sure you only extract relevant keywords, which might be useful to filter a database. Things like 'norm', 'XXX' etc. are not relevant.

Here are the descriptions for the individual categories, use them: 

{KEYWORD_TYPE_DESCRIPTIONS}

Here is the text:
"{user_prompt}"
"""

# Prompt an LLM zur Überprüfung der Kontext-Relevanz
context_relevance_prompt_template = """
This is the context:
{context}

This is the user's prompt:
{user_prompt}
Evaluate whether the context is relevant to answer the user's prompt.
Provide a grade ('relevant' or 'not relevant') and a short feedback if it is not relevant.
"""

# Prompt für SQL Query Erstellung und Anpassung
explain_query_to_llm = """
You are a tool-only SQL query generation agent.

RULES:
- NEVER respond with natural language.
- ALWAYS respond with a tool call OR with an empty assistant message ("") if no tool is applicable.
- NEVER explain, summarize, justify, or output text.
- ALWAYS follow this workflow exactly:

Database: 
The Database contains texts in context to investigations of pressure or temperature sensors or parts of it.


WORKFLOW:
1. Generate an SQL query based on the user prompt and the extracted keywords.
2. Call the runQuery tool with this SQL query.
3. If runQuery returns no instigation IDs, adjust the SQL filters and call runQuery again.
4. As soon as instigation IDs are available, call the get_SQL_Texts tool with all IDs.
5. If no results are found, try tochange the filters or reduce the filters to just relevant once so you get results. 
6. Evaluate whether the returned texts match the user prompt.
   - If NOT relevant → modify the SQL query and call runQuery again.
   - If relevant → return "" (empty assistant message).


BASE SQL STRUCTURE (must always be used, with your added WHERE filters):


-- ==============================================
-- EXAMPLE QUERY TEMPLATE FOR AGENT (Full-Text Search, no CTE, no curly braces)
-- ==============================================


SELECT DISTINCT i.id AS InstigationId
FROM Joblists jl
JOIN Instigations i ON jl.Instigationid = i.id
LEFT JOIN PurposeOfUse pou ON jl.PurposeOfUseid = pou.id
LEFT JOIN PerformanceTests pt ON jl.PerformanceTestid = pt.id
LEFT JOIN SummaryAndNotes sn ON jl.Instigationid = sn.Instigationid
LEFT JOIN Devices d ON jl.Deviceid = d.id
LEFT JOIN DeviceParameters dp ON d.id = dp.Deviceid
JOIN AdjustmentsGrouped adjg ON d.id = adjg.Deviceid
JOIN PartLists pl ON d.id = pl.Deviceid
JOIN Parts pa ON pl.Partid = pa.id
JOIN Norms n ON pt.Normid = n.id
JOIN Users u ON i.ResponsibleUserid = u.id
WHERE 
    pl.Revision = 0 
    AND pa.isToplevelPart = 1

    -- ▼▼ FULL-TEXT SEARCH FILTERS (EN + DE) ▼▼
    AND (
        CONTAINS(i.Description, '"TEXT_EN*"') OR CONTAINS(i.Description, '"TEXT_DE*"')
     OR CONTAINS(pou.Description, '"TEXT_EN*"') OR CONTAINS(pou.Description, '"TEXT_DE*"')
     OR CONTAINS(pt.Description, '"TEXT_EN*"') OR CONTAINS(pt.Description, '"TEXT_DE*"')
     ....
    )


These are relevant filters and should e used in an AND clause, combined with the ones below
    -- ▼▼ HARD FILTERS PLACEHOLDER ▼▼
    -- AND d.ProductCode = 'Productcodes like 505, 711 etc... '
    -- AND pa.OrderNumber = 'Ordernumbers like 1.4364121.001 or 1.43645822....'
    -- AND d.Manufacturer LIKE 'Huba Control, OULD, Siemens....'
    -- AND pa.ArticleNumber = '91161B1, Productcode and Art number are usually combined like this: 505.94123' 
    -- AND i.Number = '24D002, 25T010 (YY-letter-3digit number)'
    -- AND i.CustomerReferenceNumber = 'RA0256, TE1512 or any other number '
    -- AND n.norm LIKE '60068-2-2, 60050-300'
    -- AND u.FirstName = 'PERSON_FIRSTNAME' OR u.LastName = 'PERSON_LASTNAME' OR u.Initials = 'PERSON_INITIALS'
    -- AND d.Label LIKE 'custom text, usually contains some information about the DUT'
    -- AND d.Remark LIKE 'custom text, usually contains some information about the DUT'
    -- AND (adjg.InputOffset = SENSORRANGE_INPUTOFFSET OR adjg.InputFullscale = SENSORRANGE_INPUTFULLSCALE OR ...)
;




Translate the filter used here in english and german and combine them with an OR
Create clever filters with single words, use only the wordstam. Do not create to many filters. 
These filters are soft filters and should usually combined with an OR

- i.Description: Contains some overall information about the instigation
- i.Title: Title of the instigation (Use Like %xxx% for Title)
- pou.Description: Contains some information about the test-group
- pou.Title: Title of the test-group (Use Like %xxx% for Title)
- pt.Title: Contains the name of a test (Use Like %xxx% for Title)
- pt.Description: Contains information about a test
- sn.Summary: Contains results of a complete instigation or a purposeOfUse


USER PROMPT:
{user_prompt}

EXTRACTED KEYWORDS:
{keywords}


"""




elabJobsContext="""
Below you can find the raw text of reports: 
Here you can see how the texts are structured:


    

--------------------------------
{Instigation_texts}
"""

answer_Prompt_template =\
"""
You answer technical answers. You might have some context available below. 

You MUST not ask the user any follow-up questions!

Additionally you have access to tools, which 
have access to different datasources:  
-queryGlossar: Contains many internally relevant terms. Check it out if technical terms are not very clear to you.
-queryDatasheets: Contains many relevant datasheets, check it out if you need more information, use product codes like 505, 711 etc.

Answer the users question based on the given context and your retried information from the tools in a technical language.
Try to use tables where it is usefull, mention paths to documents and users which were involved

Make sure to include paths to files and user names in the answer, so the user can find the information easily.


Here is the users query:\n
{user_prompt}

Below xou can find instigation texts in the following format: 
Level 1: Instigation
The overarching container, e.g., “Development of a new sensor”
Describes the overall project or topic
Contains multiple Purpose of Use (POU)

Level 2: Purpose of Use (POU)
- Represents a test series or a specific application purpose
- Each POU follows a similar structure:
    - Title or designation of the test series
    - Description of the objective
    - List of included tests (Performance Tests)

Level 3: Performance Test (PT)
- Individual examinations or measurements within a POU
- Repetitive structure containing details such as:
    - Name or type of test
    - Objective or purpose
    - Execution / conditions
    - Result or evaluation

----------
Texts
----------
{instigation_texts}

Anwer the question in a structured way, including all datasources you have access to. Make an easy readable structure, where you clearly seperate different parts. 
"""



# Prompt für die Beantwortung der Userfrage auf Basis der Instigation-Texts
answer_from_context_prompt_template = """
You get a text with the following structure:
You may recieve some texts, which contains some results and texts from an technical instigation. 


Try to answer the users prompt. Additionally, you have access to Retrieval tool, which have access to the followig tools
Standard library: Contains relevant Standards like 60068‑2-80 Mixed Mode Vibrationen etc. Query by passing relevant keywords or phrases which might be semantic close to its text
Datasheet library: Contains many relevant datasheets, query by passing a productcode like 505, 711 etc. 

Anser the users question based on the given context, your retrieved information from the tools in a technical language. 
If datalocations are available, return them to help the user find more information. Same with people which are meantioned in the context. 

Here is the prompt: {user_prompt}
Here is the context: {instigation_texts}



"""

# Optional: Prompt für Fehler-Feedback oder Query-Optimierung
sql_feedback_prompt_template = """
The following SQL query did not seem to provide relevant data:
{sql}

Here is feedback why it's not relevant:
{feedback}

Try to adjust the filters accordingly and return a new set of data.
"""

keyword_descriptions = {
    "ProductCode": (
        "Refers to a product. Usually a 3-digit internal code.  For example 505, 211, 712 etc....)"
        "Special cases: \n"
        "- Codes starting with 5xx refer exactly to the following products and no others: 520, 528, 578, 525, 526. \n"
        "- Codes starting with 2xx refer to all products whose codes begin with 2 (e.g., 210, 211, 240, 250, etc.). \n"
        "IMPORTANT: If the code contains a dot (.), for example 505.99103, you **must split it** into two parts: "
        "1) the part before the dot is always ProductCode (505) "
        "2) the part after the dot is always Variant (99103). "
        "Do not treat the part after the dot as ProductCode. Always return both fields separately."),
    "Ordernumber": (
        "Internal order numbers, e.g., 1.4364121.001 or 1.43645822."
    ),
    "Manufacturer": (
        "The product manufacturer. Internal products come from Huba Control; other manufacturers "
        "can be Sensata, OULD, Shxieme, etc."
    ),
    "Variant": (
        "Internal Variant numbers that follow a ProductCode, always separated by a dot. "
        "Example: 505.990012 or 211.914752C. "
        "Only the part after the dot is the ArtNumber."
    ),

    "InstigationNumber": (
        "Number consisting of year, letter, and 3-digit number, e.g., 24D002 or 25T010."
    ),
    "ExternalNumber": (
        "Any other numbers, e.g., RA001546 or TE12572, which may refer to reports, customer cases, or similar."
    ),
    "Standard": (
        "Standards, e.g., 60068-2, Automotive, UL 61010-1, etc."
    ),
    "DUT_Configuration": (
        "General product-specific configurations, e.g., cell diameter, cell manufacturer, glass frit, "
        "firing temperature, etc."
    ),
    "Person": (
        "A mentioned person, either by name or initials, e.g., Vod or At."
    ),
    "ArtNumber_Internal": (
        "Internal article number consist of a 6 digit number which starts with a 1 (e.g., 123456, 105362)"
    ),
    "ArtNumber_External": (
        "All other Article-numbers of Products etc. which are not internal numbers"
    ),
    "SensorRange ": (
        "The sensorrange of a sensor usually consists of an input and output range an unit. For example 0 - 5 bar 4 - 20 mA, but the user can specify the range in varius ways. Here you can find some examples, return input and output accordingly: "
        "5 bar ratiometric --> input 0 to 5 bar, output ratiometric"
        "-50 - 50 Pa, 0 - 10 V --> input -50 to 50 Pa, output 0 to 10 V"
        "Stromtypen --> output current type"
        "etc...."
    )
}

keyWordExtractionPrompt = """
  You are a keyword extractor.
  Extract only the essential product numbers, job IDs, or unique terms from this text.
  For each extracted keyword, assign its type. 

  Make sure you only extract relevant keywords, which might be usefull to filter a database. Things like 'norm', 'XXX' etc are not relevant.

  Here are the descriptions for the individual categories, use them: 

  {keyword_descriptions}


  Here ist the Text:
  "{user_prompt}"
  
  
  
  """


queryInstigationTexts="""SELECT 
            -- WorkPackages
            wp.id AS WorkPackageId,
            wp.PurposeOfUseid AS WP_PurposeOfUseid,
            wp.PerformanceTestid AS WP_PerformanceTestid,
            wp.JobTypeid AS WP_JobTypeid,
            wp.Findings AS WP_Findings,
            wp.FunctionalState AS WP_FunctionalState,

            jt_WP.Type as wp_JobType,
            jt_JL.Type as jl_JobType,
            -- SummaryAndNotes
            sn.id AS SummaryAndNotesId,
            sn.Instigationid AS SN_Instigationid,
            sn.PurposeOfUseid AS SN_PurposeOfUseid,
            sn.Type AS SN_Type,
            sn.Summary AS SN_Summary,
            sn.DateTimeStamp AS SN_DateTimeStamp,
            sn.Report AS SN_Report,
            sn.PassFail AS SN_PassFail,
            sn.Notes AS SN_Notes,

            -- Instigation / JobLists.PurposeOfUse
            i.id AS I_Instigationid,
            i.Title AS I_Title,
            i.Description AS I_Description,
            i.DueDate AS I_DueDate,
            i.Number AS I_Number,
            i.CustomerReferenceNumber AS I_CustomerReferenceNumber,
            i.EndDate AS I_EndDate,
            i.Classification AS I_Classification,
            i.CreatorUserid AS I_CreatorUserid,
            i.ResponsibleUserid AS I_ResponsibleUserid,
            i.DateTimeStamp AS I_DateTimeStamp,
            i.InstigationTypeid  AS I_InstigationTypeid,

            -- PurposeOfUse-Tabelle (POU)
            pou.id AS PurposeOfUse_id,
            pou.Instigationid as PurposeOfUse_Instigationid,
            pou.Title AS PurposeOfUse_Title,
            pou.Description AS PurposeOfUse_Description,
            pou.DueDate AS PurposeOfUse_DueDate,
            pou.Number AS PurposeOfUse_Number,

            -- PerformanceTest (PT)
            pt.id AS PT_PerformanceTestid,
            pt.Instigationid AS PT_Instigationid,
            pt.Title AS PT_Title,
            pt.Normid AS PT_Normid,
            pt.Description AS PT_Description,
            pt.TestLevel AS PT_TestLevel,
            pt.EstimatedDuration AS PT_EstimatedDuration,
            pt.Accuracy AS PT_Accuracy,
            pt.FunctionalStatus AS PT_FunctionalStatus,
            pt.Criteria AS PT_Criteria,
            pt.TestDescription AS PT_TestDescription,
            
            -- Users
            u.FirstName AS User_FirstName,
            u.LastName AS User_LastName,
            u.Initials AS User_Initials
    
        FROM [ELabJobs].[dbo].[JobLists] jl
        LEFT JOIN [ELabJobs].[dbo].[WorkPackages] wp
            ON jl.[PurposeOfUseid] = wp.[PurposeOfUseid]
           AND jl.[PerformanceTestid] = wp.[PerformanceTestid]
        LEFT JOIN [ELabJobs].[dbo].[SummaryAndNotes] sn
            ON jl.[Instigationid] = sn.[Instigationid]
           AND jl.[PurposeOfUseid] = sn.[PurposeOfUseid]
        LEFT JOIN [ELabJobs].[dbo].[Instigations] i
            ON jl.[Instigationid] = i.id
        LEFT JOIN [ELabJobs].[dbo].[PerformanceTests] pt
            ON jl.[PerformanceTestid] = pt.id
           AND jl.[Instigationid] = pt.[Instigationid]
        LEFT JOIN [ELabJobs].[dbo].[PurposeOfUse] pou
            ON jl.[PurposeOfUseid] = pou.id

            Left JOIN [ELabJobs].[dbo].[JobType] jt_WP
            ON wp.JobTypeid = jt_WP.id
                
        Left JOIN [ELabJobs].[dbo].[Users] u
            ON i.ResponsibleUserid = u.id 
            

            Left JOIN [ELabJobs].[dbo].JobType jt_JL
            ON jl.JobTypeid = jt_JL.id
            WHERE i.id = {instigationid}
            """