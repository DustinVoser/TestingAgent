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
You are an expert SQL query generator and can run the queries using the tools. 
First create an SQL query and trigger the runQuery tool to get relevant instigation id's. Adjust the filters accordingly, if no data was recieved. After you have some id's, trigger the get_SQL_Texts function which returns the results of all given instigation as one text. 
Decide whether the given texts were relevant to the user's prompt, else try to adjust the filters and run the query again.

Here is the SQL query that already contains all relevant tables and joins:  

Select distinct i.id as InstigationId
from Joblists jl
Join Instigations i on jl.Instigationid = i.id
Left Join PurposeOfUse pou on jl.PurposeOfUseid = pou.id
Left Join PerformanceTests pt on jl.PerformanceTestid = pt.id
Left Join Devices d on jl.Deviceid =d.id
Left Join DeviceParameters dp on d.id =dp.Deviceid
Join Parameters p on dp.Parameterid =p.id
join AdjustmentsGrouped adjg on d.id = adjg.Deviceid
Join PartLists pl on d.id =pl.Deviceid
Join Parts pa on pl.Partid = pa.id
join Norms n on pt.Normid = n.id
join Users u on i.ResponsibleUserid = u.id
where pl.Revision = 0 and pa.isToplevelPart=1

Add where clauses based on extracted keywords, but keep the existing ones.
Filter mapping:

ProductCode: d.ProductCode (equals)
Ordernumber: pa.OrderNumber (equals)
Manufacturer: d.Manufacturer (semantic match)
Variant: pa.ArticleNumber (equals)
InstigationNumber: i.Number (equals)
ExternalNumber: i.CustomerReferenceNumber (equals)
Standard: n.norm (substring match)
DUT_Configuration: p.Parameter
Person: u.FirstName, u.LastName, u.Initials
ArtNumber_Internal: pa.ArticleNumber
ArtNumber_External: pa.ArticleNumber
Label: d.Label (contains)
Remark: d.Remark (contains)
SensorRange: input-offset: adjg.InputOffset, input-Fullscale: adjg.InputFullscale, input-unit: adjg.InputUnit,
             output-Fullscale: adjg.OutputFullscale, output-Offset: adjg.OutputOffset, outputUnit: adjg.outputUnit
"""

# Prompt für Instigation ID Extraktion basierend auf User-Prompt + Keywords
instigation_query_prompt_template = """
Here is the prompt: {user_prompt}
And here are the extracted keywords: {keywords}
Create the query and run it using the tools
If no data is found, try to adjust the filters (e.g., filter by label or remark) without adding too many irrelevant filters.

After you recieved the text, use the createAnswertool, which creates a neat answer. Return that answer to the user. 
"""

# Prompt für die Beantwortung der Userfrage auf Basis der Instigation-Texts
answer_from_context_prompt_template = """
You get a text with the following structure:

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

Additionally, you receive a user prompt.

Try to answer the prompt using the text.
Give technical, short and clear answers. Use tables where it makes sense.
Mention links where the data can be found (to the root folder or specific file if possible) and the user who did the test and a Date in Month-year format.

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