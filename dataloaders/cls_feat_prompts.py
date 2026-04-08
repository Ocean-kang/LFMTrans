

prompts_1 = [
    ['A photo of a [cls]',
     'This is a photo of a [cls]',
     'A photo of a small [cls]',
     'A photo of a medium [cls]',
     'A photo of a large [cls]',
     'This is a photo of a small [cls]',
     'This is a photo of a medium [cls]',
     'This is a photo of a large [cls]',
     'There is a [cls] in the scene',
     'There is the [cls] in the scene',
     'A photo of a [cls] in the scene',
     'There is a small [cls] in the scene',
     'There is a medium [cls] in the scene',
     'There is a large [cls] in the scene',
     ]
]

prompts_2 = [
    ['Describe the [cls] in detail, highlighting its unique visual features.',
     'What are the most distinctive visual characteristics of a [cls]?',
     'Show me the key visual features that distinguish a [cls] from others.',
     'What makes a [cls] visually recognizable?',
     'Provide a detailed description of the [cls]\'s appearance, focusing on its most distinctive aspects.'
     ]
]
prompts_3 = [
    ['Describe the key visual features of [cls] that make it different from other categories.',
     'What are the main visual characteristics of [cls] that distinguish it from similar categories?',
     'How can [cls] be recognized visually? Focus on its shape, color, and texture.',
     'What visual features set [cls] apart from other categories in the image?',
     'Describe the unique visual traits of [cls] that help it stand out from others.'
     ]
]
prompts_4 = [
    [
        'Please describe the key visual characteristics of [cls]. Focus on the attributes that make it distinct from other similar categories. Explain the defining features such as shape, texture, or color that set [cls] apart.',
        'Given the category [cls], identify the primary visual traits that differentiate it from other semantic categories. What specific visual cues—like shape, size, or structure—can be used to recognize [cls] in a set of images?',
        'For the category [cls], describe its distinguishing visual features. Highlight any attributes such as color, texture, size, or form that make [cls] unique compared to other visually similar categories.',
        'Given the semantic category [cls], explain how its visual representation differs from other categories in terms of shape, proportion, texture, or color. What visual feature is most critical for distinguishing [cls] from similar categories?',
        'For [cls], identify and explain the key visual elements that make it recognizable. Consider attributes like shape, structure, color, or patterns, and describe how these visual characteristics differentiate [cls] from other categories in the image set.'
        ]
]
prompts_5 = [
    [
        "What is the key characteristic that distinguishes [cls] from its visual analogues in the image set?",
        "What sets [cls] apart from other categories with similar visual features in the image set?",
        "Can you identify the unique visual cue that defines [cls] in the image set?",
        "What is the primary visual characteristic that distinguishes [cls] from other categories with similar appearances in the image set?",
        "What visual attribute or feature is most distinctive about [cls] compared to its visual analogues in the image set?"
    ]
]

# A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties
prompts_6 = [
    ["Please make the descriptions to have a similar level of detail and a consistent type of information, "
     "which should be beneficial for clustering and machine learning applications. "
     "Each description will follow a structured format:\n"
     "- Start with a general description of the object or scene. \n"
     "- Describe the shape, orientation, and primary physical characteristics. \n"
     "- Mention the material, texture, or typical colors. \n"
     "- Note common features or elements associated with the object or scene. \n"
     "- End with possible additional details, variations, or environmental context. \n"
     "Please describe [cls]:"]
]


# yiyan
prompts_8 = [
    [
        f"Describe the shape, color palette, and typography that would best represent the [cls] visually.",
        "How would you layout the [cls] in a design to make it stand out?",
        "What visual elements would you include to convey the essence of the [cls]?",
    ]
]

prompts_9 = [
    [
        f"How would you frame the shot to capture the essence of the [cls]'s visual identity?",
        "What lighting techniques would you use to highlight the [cls]'s features?",
        "Describe the atmosphere and setting that would complement the [cls]'s visual presence.",
    ]
]


prompts_10 = [
    [
        f"Use imagery to paint a picture of the [cls]'s visual appeal.",
        "Compare the [cls] to nature or another artwork to draw parallels in visual beauty.",
        "How does the [cls]'s visual form inspire emotions or thoughts in you?",
    ]
]


prompts_11 = [
    [
        f"Describe the brushstrokes and colors you would use to capture the [cls]'s texture and hue.",
        "What composition techniques would you employ to draw attention to the [cls]'s most striking features?",
        "How would you blend reality with abstraction in your depiction of the [cls]?",
    ]
]


prompts_12 = [
    [
        f"Detail the visual elements that would make the [cls] the focal point of the VR space.",
        "How would you use sound and lighting to enhance the [cls]'s visual impact?",
        "What interactive features would you incorporate to allow users to explore the [cls]'s visual details?",
    ]
]

prompts_13 = [
    [
        f"Discuss the props and backgrounds you would use to complement the [cls]'s visual appeal.",
        "What angles and perspectives would you choose to highlight the [cls]'s unique visual characteristics?",
        "How would you use post-production techniques to enhance the [cls]'s visual presentation?",
    ]
]
prompts_14 = [
    [
        f"Detail the textures, shaders, and animations that would make the [cls] visually appealing in-game.",
        "How would you design its interaction mechanics to align with the [cls]'s visual identity?",
        "What visual cues would you use to signal the [cls]'s function or role within the game world?",
    ]
]
prompts_15 = [
    [
        f"Translate the visual language of the [cls] into fashion terms, such as colors, patterns, and materials.",
        "How would you incorporate the [cls]'s visual essence into a wearable design?",
        "Describe the target audience and mood you envision for the [cls]-inspired fashion piece.",
    ]
]

# deepseek
prompts_16 = [[f"Please provide a detailed description of the visual features of [cls] including its color, shape, texture, size, and other relevant attributes.",]]
prompts_17 = [[f"Describe the visual contrast features of [cls] compared to other objects, including differences in color, shape, texture, size, and other relevant attributes.",]]
prompts_18 = [[f"Describe the visual features of [cls] within a specific scene, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_19 = [[f"Describe the visual features of [cls] based on its functionality or purpose, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_20 = [[f"Describe the visual features of [cls] based on its material composition, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_21 = [[f"Describe the visual features of [cls] in a way that evokes specific emotions or associations, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_22 = [[f"Describe the visual features of [cls] based on its cultural significance or symbolism, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_23 = [[f"Describe the visual features of [cls] in motion or in a dynamic context, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_24 = [[f"Describe the visual features of [cls] in an abstract or metaphorical way, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_25 = [[f"Describe the visual features of [cls] based on scientific principles or observations, including its color, shape, texture, size, and other relevant attributes.",]]
prompts_26 = [
    [
        "What are the most distinctive features of [cls]? How does it differ from others?",
        "Can you categorize [cls] based on its shape, color, or function? How does it fit into the broader category of [cls]?",
        "How does [cls] compare to others in terms of size, shape, or functionality? What are the key differences?",
        "What is the primary purpose of [cls]? How does it interact with other objects or the environment?",
        "Where would you typically find [cls]? How does it relate to its surroundings or other objects in the scene?"
    ]
]
# prompts_0 = [
#     prompts_18[0] + prompts_19[0] + prompts_20[0]
# ]

prompts_0 = [
    prompts_2[0] +
    prompts_3[0] +
    prompts_4[0] +
    prompts_5[0] +
    prompts_6[0] +
    prompts_8[0] +
    prompts_9[0] +
    prompts_10[0] +
    prompts_11[0] +
    prompts_12[0] +
    prompts_13[0] +
    prompts_14[0] +
    prompts_15[0] +
    prompts_16[0] +
    prompts_17[0] +
    prompts_18[0] +
    prompts_19[0] +
    prompts_20[0] +
    prompts_21[0] +
    prompts_22[0] +
    prompts_23[0] +
    prompts_24[0] +
    prompts_25[0] +
    prompts_26[0]
]

prompts_27 = [
    [
        "Describe the shape of a [cls].",
        "What is the typical size of a [cls]?",
        "Is a [cls] usually heavy or light?",
        "What is the texture of a [cls]?",
        "Is a [cls] typically smooth or rough?",
        "What is the color of a [cls]?",
        "Is a [cls] transparent, opaque, or translucent?",
        "Does a [cls] have a distinct smell?",
        "Is a [cls] usually cold or warm to the touch?",
        "What is the weight distribution of a [cls]?",
        "What is the primary function of a [cls]?",
        "How is a [cls] typically used?",
        "Is a [cls] used for a specific task or multiple tasks?",
        "Does a [cls] have a secondary function or purpose?",
        "Is a [cls] essential for a particular activity or process?",
        "Can a [cls] be used in different environments or settings?",
        "Is a [cls] limited to a specific industry or profession?",
        "Can a [cls] be used by individuals or groups?",
        "How does a [cls] differ from its analogue?",
        "What are the key differences between a [cls] and its closest relative?",
        "Is a [cls] more advanced or primitive than its analogue?",
        "Does a [cls] have any unique features that distinguish it from its analogue?",
        "Can a [cls] be used in situations where its analogue cannot?",
        "Is a [cls] more durable or fragile than its analogue?",
        "Where is a [cls] typically found?",
        "Does a [cls] occur naturally or is it manufactured?",
        "Is a [cls] found in a specific climate or environment?",
        "Can a [cls] be found in different parts of the world?",
        "Is a [cls] a common or rare occurrence?",
        "Does a [cls] have a specific habitat or ecosystem?",
        "How does a [cls] interact with its surroundings?",
        "Does a [cls] have any specific behaviors or patterns?",
        "Is a [cls] social or solitary?",
        "Can a [cls] be trained or domesticated?",
        "Does a [cls] have any notable interactions with other objects or creatures?",
        "Is a [cls] known for its intelligence or problem-solving abilities?",
        "What is the origin of a [cls]?",
        "How has a [cls] evolved over time?",
        "Is a [cls] a recent invention or has it been around for centuries?",
        "Does a [cls] have any historical significance or cultural importance?",
        "Can a [cls] be traced back to a specific civilization or era?",
        "Has a [cls] undergone significant changes or updates over the years?",
        "What sets a [cls] apart from other objects?",
        "Does a [cls] have any unique features or characteristics?",
        "Is a [cls] known for its aesthetic appeal or design?",
        "Can a [cls] be customized or modified?",
        "Does a [cls] have any notable accessories or attachments?",
        "Is a [cls] a collectible or rare item?",
        "What are common misconceptions about a [cls]?",
        "Are there any myths or legends surrounding a [cls]?",
        "Does a [cls] have any surprising or unexpected properties?",
        "Are there any common mistakes people make when using a [cls]?",
        "Can a [cls] be used in ways that are not immediately apparent?",
        "How does a [cls] compare to other objects in its class?",
        "Is a [cls] more or less effective than its competitors?",
        "Does a [cls] have any notable advantages or disadvantages?",
        "Can a [cls] be used in combination with other objects?",
        "Is a [cls] a replacement for an older or outdated technology?",
        "How should a [cls] be maintained or cared for?",
        "Are there any specific cleaning or storage requirements?",
        "Does a [cls] require regular maintenance or upkeep?",
        "Can a [cls] be repaired or serviced?",
        "Is a [cls] prone to damage or wear and tear?",
        "Does a [cls] have any specific handling or transportation requirements?",
    ]
]

prompts_28 = [
    [
        "Describe the appearance of a [cls].",
        "What does a [cls] look like?",
        "How would you describe the appearance of a [cls]?",
        "Give me a detailed description of a [cls].",
        "What are the main features of a [cls]?",
        "Describe the shape, size, and color of a [cls].",
        "How does a [cls] differ from its analogues?",
        "What does a [cls] look like in a [surrounding environment]?",
        "Describe the texture and material of a [cls].",
        "What are the distinguishing features of a [cls]?",
        "Give me a description of the appearance of a [cls] in a [specific context].",
        "How would you recognize a [cls] from a distance?",
        "What are the key characteristics of a [cls]?",
        "Describe the appearance of a [cls] in a [specific lighting condition].",
        "How does a [cls] relate to its surroundings?",
        "What are the main visual features of a [cls]?",
        "Describe the appearance of a [cls] in a [specific cultural or historical context].",
        "How would you describe the overall appearance of a [cls]?",
        "What are the notable features of a [cls]?",
        "Give me a detailed description of the appearance of a [cls] in a [specific environment].",
        "How does a [cls] stand out in a [surrounding environment]?",
        "What are the distinguishing physical characteristics of a [cls]?",
        "Describe the appearance of a [cls] in a [specific situation].",
        "How would you recognize a [cls] by its appearance?",
        "What are the main visual cues of a [cls]?",
        "Describe the appearance of a [cls] in a [specific industry or field].",
        "How does a [cls] relate to its purpose?",
        "What are the notable visual features of a [cls]?",
        "Give me a description of the appearance of a [cls] in a [specific setting].",
        "How would you describe the overall appearance of a [cls] in a [specific context]?",
        "What are the key visual characteristics of a [cls]?",
        "Describe the appearance of a [cls] in a [specific style or design].",
        "How does a [cls] differ from other [cls]?",
        "What are the main features of a [cls] that make it recognizable?",
        "Give me a detailed description of the appearance of a [cls] in a [specific environment].",
        "How would you recognize a [cls] by its shape?",
        "What are the notable physical characteristics of a [cls]?",
        "Describe the appearance of a [cls] in a [specific scenario].",
        "How does a [cls] relate to its function?",
        "What are the main visual cues of a [cls] that make it recognizable?",
        "Describe the appearance of a [cls] in a [specific industry or sector].",
        "How does a [cls] stand out in a [specific context]?",
        "What are the key physical characteristics of a [cls]?",
        "Give me a description of the appearance of a [cls] in a [specific setting].",
        "How would you describe the overall appearance of a [cls]?",
        "What are the notable visual features of a [cls] that make it recognizable?",
        "Describe the appearance of a [cls] in a [specific style or aesthetic].",
        "How does a [cls] differ from its analogues in terms of appearance?",
        "What are the main features of a [cls] that make it unique?",
        "Give me a detailed description of the appearance of a [cls] in a [specific environment].",
        "How would you recognize a [cls] by its color?",
        "What are the notable physical characteristics of a [cls] that make it recognizable?",
        "Describe the appearance of a [cls] in a [specific scenario].",
        "How does a [cls] relate to its purpose in terms of appearance?",
        "What are the main visual cues of a [cls] that make it recognizable?",
        "Describe the appearance of a [cls] in a [specific industry or sector].",
        "How does a [cls] stand out in a [specific context]?",
        "What are the key physical characteristics of a [cls] that make it recognizable?",
        "Give me a description of the appearance of a [cls] in a [specific setting].",
        "How would you describe the overall appearance of a [cls]?",
        "What are the notable visual features of a [cls] that make it unique?",
        "Describe the appearance of a [cls] in a [specific style or aesthetic].",
        "How does a [cls] differ from its analogues in terms of appearance?",
        "What are the main features of a [cls] that make it recognizable?",
    ]
]

prompts_29 = [
    [
        "Describe the primary shape and silhouette of [cls] in detail.",
        "What dominant colors and color gradients are characteristic of [cls]?",
        "How does the texture of [cls] differ from its closest analogues?",
        "List the key structural components or parts that define [cls].",
        "What unique patterns, markings, or symbols are typically found on [cls]?",
        "How does the size and proportion of [cls] aid in distinguishing it from similar objects?",
        "Describe the surface finish of [cls] (e.g., glossy, matte, rough).",
        "What are the most common materials used for [cls], and how do they affect its appearance?",
        "How do lighting conditions (e.g., shadows, reflections) alter the visual perception of [cls]?",
        "What visual features differentiate [cls] in motion versus at rest?",
        "Are there any protrusions, indentations, or openings on [cls]? Describe their placement.",
        "How does [cls] contrast with its typical environment or background?",
        "What are the symmetry or asymmetry characteristics of [cls]?",
        "Describe the edges, contours, and angles of [cls].",
        "What accessories or attachments are commonly associated with [cls], and how do they look?",
        "How do wear, aging, or environmental exposure typically alter [cls]’s appearance?",
        "What visual cues distinguish authentic [cls] from counterfeit versions?",
        "Describe any logos, labels, or text prominently displayed on [cls].",
        "How does the interior of [cls] differ visually from its exterior?",
        "What is the typical arrangement or layout of components on [cls]?",
        "How do seasonal or contextual variations affect the appearance of [cls]?",
        "What visual traits remain consistent across all variants of [cls]?",
        "Describe the optical properties of [cls] (e.g., transparency, reflectivity).",
        "How do joints, seams, or fasteners on [cls] contribute to its visual identity?",
        "What repetitive or geometric patterns are unique to [cls]?",
        "How does [cls]’s weight distribution influence its physical form?",
        "What visual elements indicate the primary function of [cls]?",
        "Describe the visual density (bulky vs. streamlined) of [cls].",
        "How do interactive elements (e.g., buttons, handles) on [cls] appear?",
        "What are the common visual imperfections or artifacts seen on [cls]?",
        "How does [cls]’s appearance differ when viewed from various angles (front, side, top)?",
        "What color contrasts or combinations are iconic to [cls]?",
        "Describe the proportionality between different sections of [cls].",
        "How do modular or detachable parts of [cls] affect its overall look?",
        "What visual traits help identify [cls] in low-light conditions?",
        "How does [cls]’s packaging or container influence its recognizability?",
        "What are the most minimal visual features sufficient to recognize [cls]?",
        "Describe the interplay of light and shadow on [cls]’s surface.",
        "How do brand-specific design elements make [cls] visually distinct?",
        "What visual traits indicate the age or condition of [cls]?",
        "How does [cls]’s design balance aesthetics and functionality visually?",
        "Describe the visual weight distribution (e.g., top-heavy) of [cls].",
        "What decorative elements (e.g., stickers, engravings) are typical for [cls]?",
        "How does the thickness or thinness of [cls]’s materials impact its appearance?",
        "What visual features distinguish [cls]’s front view from its rear view?",
        "How do environmental interactions (e.g., dirt, weathering) alter [cls]’s look?",
        "Describe the visual hierarchy of [cls]’s components (most to least prominent).",
        "What are the common color transitions or gradients on [cls]?",
        "How does [cls]’s surface interact with moisture (e.g., water beading)?",
        "What visual traits signal [cls]’s intended use or context?",
        "How do the edges of [cls] (sharp, rounded) contribute to its recognition?",
        "Describe any vents, grilles, or perforations on [cls] and their layout.",
        "What visual characteristics differentiate [cls]’s daytime vs. nighttime appearance?",
        "How do tactile features (e.g., grooves, ridges) of [cls] appear visually?",
        "What is the role of negative space in [cls]’s design?",
        "How does [cls]’s scale relative to human users affect its visual traits?",
        "Describe the visual rhythm created by repeating elements on [cls].",
        "What visual traits help distinguish [cls] in cluttered environments?",
        "How do [cls]’s movable parts alter its appearance when activated?",
        "What is the role of typography or fonts in [cls]’s visual identity?",
        "Describe the interplay of curves and straight lines in [cls]’s design.",
        "How does [cls]’s opacity or translucency affect its recognizability?",
        "What visual traits make [cls] stand out in aerial or top-down views?",
        "How do [cls]’s color and texture mimic or contrast with natural elements?"
    ]
]

prompts_30 = [
    [
        "Describe the [cls] in detail, focusing on its shape, size, and distinctive features.",
        "Provide a comprehensive description of the [cls], highlighting its key visual characteristics.",
        "Explain the appearance of the [cls], emphasizing its unique attributes and overall form.",
        "Detail the visual aspects of the [cls], including its dimensions and notable features.",
        "Outline the main visual features of the [cls], describing its shape and size.",
        "Characterize the [cls] by describing its prominent visual traits and overall structure.",
        "Depict the [cls] by detailing its shape, size, and any distinguishing features.",
        "Illustrate the appearance of the [cls], focusing on its key visual elements.",
        "Provide a detailed account of the [cls]'s appearance, emphasizing its unique features.",
        "Describe the [cls] in terms of its shape, size, and any notable visual characteristics.",
        "Explain the visual features of the [cls], highlighting its distinctive attributes.",
        "Detail the [cls]'s appearance, focusing on its shape, size, and key features.",
        "Characterize the [cls] by describing its main visual elements and overall form.",
        "Depict the [cls] by outlining its shape, size, and distinguishing features.",
        "Illustrate the [cls]'s appearance, emphasizing its unique visual traits.",
        "Provide a comprehensive description of the [cls], focusing on its key visual aspects.",
        "Describe the [cls] in detail, highlighting its shape, size, and distinctive features.",
        "Explain the appearance of the [cls], emphasizing its unique attributes and overall form.",
        "Detail the visual aspects of the [cls], including its dimensions and notable features.",
        "Outline the main visual features of the [cls], describing its shape and size.",
        "Characterize the [cls] by describing its prominent visual traits and overall structure.",
        "Depict the [cls] by detailing its shape, size, and any distinguishing features.",
        "Illustrate the appearance of the [cls], focusing on its key visual elements.",
        "Provide a detailed account of the [cls]'s appearance, emphasizing its unique features.",
        "Describe the [cls] in terms of its shape, size, and any notable visual characteristics.",
        "Explain the visual features of the [cls], highlighting its distinctive attributes.",
        "Detail the [cls]'s appearance, focusing on its shape, size, and key features.",
        "Characterize the [cls] by describing its main visual elements and overall form.",
        "Depict the [cls] by outlining its shape, size, and distinguishing features.",
        "Illustrate the [cls]'s appearance, emphasizing its unique visual traits.",
        "Provide a comprehensive description of the [cls], focusing on its key visual aspects.",
        "Describe the [cls] in detail, highlighting its shape, size, and distinctive features.",
        "Explain the appearance of the [cls], emphasizing its unique attributes and overall form.",
        "Detail the visual aspects of the [cls], including its dimensions and notable features.",
        "Outline the main visual features of the [cls], describing its shape and size.",
        "Characterize the [cls] by describing its prominent visual traits and overall structure.",
        "Depict the [cls] by detailing its shape, size, and any distinguishing features.",
        "Illustrate the appearance of the [cls], focusing on its key visual elements.",
        "Provide a detailed account of the [cls]'s appearance, emphasizing its unique features.",
        "Describe the [cls] in terms of its shape, size, and any notable visual characteristics.",
        "Explain the visual features of the [cls], highlighting its distinctive attributes.",
        "Detail the [cls]'s appearance, focusing on its shape, size, and key features.",
        "Characterize the [cls] by describing its main visual elements and overall form.",
        "Depict the [cls] by outlining its shape, size, and distinguishing features.",
        "Illustrate the [cls]'s appearance, emphasizing its unique visual traits.",
        "Provide a comprehensive description of the [cls], focusing on its key visual aspects.",
        "Describe the [cls] in detail, highlighting its shape, size, and distinctive features.",
        "Explain the appearance of the [cls], emphasizing its unique attributes and overall form.",
        "Detail the visual aspects of the [cls], including its dimensions and notable features.",
        "Outline the main visual features of the [cls], describing its shape and size.",
        "Characterize the [cls] by describing its prominent visual traits and overall structure.",
        "Depict the [cls] by detailing its shape, size, and any distinguishing features.",
        "Illustrate the appearance of the [cls], focusing on its key visual elements.",
        "Provide a detailed account of the [cls]'s appearance, emphasizing its unique features.",
        "Describe the [cls] in terms of its shape, size, and any notable visual characteristics.",
        "Explain the visual features of the [cls], highlighting its distinctive attributes.",
        "Detail the [cls]'s appearance, focusing on its shape, size, and key features.",
        "Characterize the [cls] by describing its main visual elements and overall form.",
        "Depict the [cls] by outlining its shape, size, and distinguishing features.",
        "Illustrate the [cls]'s appearance, emphasizing its unique visual traits.",
        "Provide a comprehensive description of the [cls], focusing on its key visual aspects.",
        "Describe the [cls] in detail, highlighting its shape, size, and distinctive features.",
        "Explain the appearance of the [cls], emphasizing its unique attributes and overall form.",
        "Detail the visual aspects of the [cls], including its dimensions and notable features."
    ]
]

prompts_31 = [
    [
        "Describe the shape, size, and color of [cls].",
        "What are the primary physical features of [cls]?",
        "How would you describe the texture and surface of [cls]?",
        "What are the defining characteristics of [cls]'s appearance?",
        "Describe the structure and form of [cls] in detail.",
        "What is the overall color pattern of [cls]?",
        "How does the surface texture of [cls] contribute to its appearance?",
        "What shapes or geometric features stand out on [cls]?",
        "Describe the silhouette or outline of [cls].",
        "How do the proportions of [cls] compare to its size?",
        "What makes the texture of [cls] distinctive?",
        "How is the surface of [cls] reflective or matte?",
        "Does [cls] have any noticeable curves or sharp edges?",
        "How would you describe the finish of [cls] (smooth, rough, polished, etc.)?",
        "What visible patterns or markings does [cls] have?",
        "Describe the key visual features that define the front view of [cls].",
        "How would you describe [cls] from a top-down perspective?",
        "What are the most prominent details of [cls] when viewed from the side?",
        "Are there any significant visual features on the edges or corners of [cls]?",
        "What color(s) dominate the appearance of [cls]?",
        "What distinguishes the texture of [cls] from other objects?",
        "How would you describe the combination of colors on [cls]?",
        "What does the surface of [cls] look like up close?",
        "Are there any specific patterns or designs on [cls]?",
        "Describe the size and proportions of [cls].",
        "How do the lighting conditions affect the appearance of [cls]?",
        "What distinguishes the shape of [cls] from other similar objects?",
        "How does [cls] appear from different angles?",
        "What is the most noticeable visual feature of [cls]?",
        "How does the lighting highlight specific features of [cls]?",
        "What features of [cls] are most prominent in a natural environment?",
        "Does [cls] have any reflective surfaces or gloss?",
        "Describe the angles or edges that make [cls] stand out.",
        "How do the materials of [cls] influence its appearance?",
        "What visual features help you identify [cls] in a crowd of similar objects?",
        "How would you describe the overall texture and finish of [cls]?",
        "Are there any visible marks or imperfections on the surface of [cls]?",
        "What details about [cls]'s shape are critical for recognition?",
        "Describe the front-facing view of [cls].",
        "How do the color gradients or transitions appear on [cls]?",
        "What kind of light reflections or shadows can be seen on [cls]?",
        "How does the surface of [cls] interact with light?",
        "What specific visual patterns or contrasts can be seen on [cls]?",
        "How does the texture of [cls] change under different lighting conditions?",
        "What is the overall silhouette of [cls] from a distance?",
        "What unusual shapes or features appear on [cls]?",
        "How does [cls] contrast with its background in the environment?",
        "What shape or design elements distinguish [cls] from other objects nearby?",
        "How is [cls]'s surface decorated or textured?",
        "Does [cls] have any recognizable features that indicate its function or purpose?",
        "What architectural or sculptural aspects can be found on [cls]?",
        "How would you describe the relationship between different parts of [cls] in terms of size and proportion?",
        "What visual cues tell you what [cls] is used for?",
        "How does the composition of [cls] influence its visual presence in a scene?",
        "What specific markings or details make the surface of [cls] stand out?",
        "How do [cls]'s features compare when viewed up close versus from afar?",
        "What part of [cls]'s shape stands out the most?",
        "How does the texture of [cls] vary between different parts?",
        "What features would help you distinguish [cls] from similar objects?",
        "How does the texture of [cls] impact the perception of its size or shape?",
        "What other objects or environments enhance the visual recognition of [cls]?",
        "How would you describe [cls] in terms of symmetry or asymmetry?",
        "What shapes or contours are most important when recognizing [cls] in real-world scenarios?",
        "How would you describe the surface material and its reflective qualities on [cls]?"
    ]
]

prompts_32 = [
    [
        "Describe the typical color distribution and surface reflectance characteristics of [cls] in three sentences",
        "Explain the overall shape contour features of [cls] from different viewing angles",
        "Detail the visual manifestations of [cls]'s surface material (smoothness/texture/light transmission)",
        "What special visual characteristics does [cls] exhibit under intense lighting?",
        "List the primary components of [cls] and their spatial arrangement",
        "Describe proportional relationships between [cls]'s components",
        "Explain structural features of [cls]'s critical connection points",
        "Analyze [cls]'s symmetry characteristics (perfect/approximate/asymmetrical)Describe three most common environmental contexts for [cls]",
        "Explain [cls]'s spatial positioning characteristics in typical usage scenarios",
        "What objects frequently co-occur with [cls] in visual scenes?",
        "How does [cls]'s appearance vary under different weather conditions?Based on [cls]'s function, describe visual features of its moving parts",
        "Describe typical morphological changes of [cls] during operation",
        "Explain visual manifestations of functional traces on [cls]'s surface",
        "Detail visual characteristics of [cls]'s human interaction areasDescribe [cls]'s detailed visual features when observed from 10cm distance",
        "Explain [cls]'s holistic visual characteristics at 1-meter viewing distance",
        "Describe contour features of [cls] from a top-down perspective",
        "Explain structural characteristics of [cls]'s underside when viewed from belowDescribe [cls]'s visual morphology during typical motion states",
        "Explain light-shadow variation patterns when [cls] is in motion",
        "Describe relative positional changes between [cls]'s components during movement",
        "What are the key visual differences between [cls]'s static and dynamic states?",
        "Detail [cls]'s surface reflectance properties (specular/diffuse)",
        "Explain transparency/opacity characteristics of [cls]'s material",
        "Describe orientation and density patterns of [cls]'s surface textures",
        "What are typical visual manifestations of wear patterns on [cls]?",
        "Compare three significant visual differences between [cls] and similar objects",
        "Describe key visual identifiers distinguishing [cls] from other categories",
        "Explain color stability of [cls] under varying lighting conditions",
        "Compare surface characteristics of [cls] in new vs. aged conditions",
    ]
]

prompts_33 = [
    [
        "Describe the fundamental geometric shape of [cls] in 3D space (e.g., sphere/cube/cylinder combinations)",
        "List the most prominent texture characteristics on [cls] surface (e.g., smooth/striped/concave-convex)",
        "Specify spatial arrangement relationships between [cls] components (e.g., symmetrical distribution/linear alignment)",
        "Describe light-dark transition characteristics of [cls] under natural illumination",
        "Indicate color distribution patterns across [cls] parts (e.g., gradient/color block combinations)",
        "Analyze light reflection properties of [cls] materials (e.g., matte/specular reflection)",
        "Explain key functional components of [cls] in typical usage scenarios",
        "List motion patterns of [cls] operating parts (e.g., rotary button/sliding switch)",
        "Clarify physical principles enabling [cls] core functions (e.g., lever principle/fluid dynamics)",
        "Describe common viewing angle ranges for human observation of [cls] (e.g., 45° top view/eye-level view)",
        "Specify spatial occupancy characteristics of [cls] in standard usage scenarios",
        "Analyze proportional relationships between [cls] components (e.g., 1:2 aspect ratio)",
        "Describe relative size relationships between [cls] and auxiliary objects (e.g., slightly larger than A4 paper)",
        "Analyze light transmittance of [cls] primary materials (e.g., translucent/opaque)",
        "List visual contrast effects produced by [cls] material combinations",
        "Describe morphological changes of [cls] in motion (e.g., folding expansion/component displacement)",
        "Specify visual indicators during [cls] functional activation (e.g., LED blinking/mechanical part movement)",
        "Analyze spatial trajectory characteristics during [cls]-user interaction (e.g., arc sliding/linear push-pull)",
        "Describe light reflection patterns of [cls] in standard environments",
        "Describe edge characteristics of [cls] core components (e.g., rounded corner/sharp edge)",
        "Specify geometric features of [cls] interface components (e.g., USB port's rectangular groove)",
        "List connection structures of [cls] detachable parts (e.g., snap-fit/screw-thread type)",
        "Describe visual characteristics of [cls] in abnormal working states (e.g., warning light on/component misalignment)",
        "Specify indication features when [cls] power is depleted (e.g., screen off/mechanical part drooping)",
        "Describe overall contour characteristics of [cls] at macro scale",
        "List surface details of [cls] at micro scale (e.g., anti-slip pattern/brand logo)"
    ]
]


system_msg_1 = f"When given an entity name [cls] from user, " \
               f"you should always response with the following pattern: " \
               f"'[pattern]' without any other additional word."
system_msg_2 = f"When given an entity name [cls] from user, " \
               f"you should respond according to the following question: " \
               f"'[prompt]' within 128 words."
system_msg_3 = f"You will be given a category name, [cls]. " \
               f"Describe the key visual features that make [cls] distinct from other categories. " \
               f"Focus on shape, color, size, and texture. Responses were limited to 128 words."
system_msg_4 = f"You will be provided with a category name, denoted as [cls], " \
               f"which represents a specific semantic category (e.g., 'cat', 'bicycle'). " \
               f"Based on this category, your task is to identify and describe the key visual " \
               f"characteristics that distinguish [cls] from other categories. " \
               f"Focus on aspects such as shape, texture, color, size, and structure that are " \
               f"unique to [cls] and make it easily recognizable compared to similar categories. " \
               f"Responses were limited to 128 words."
system_msg_5 = f"SEMANTIC_ENTITY_DESCRIPTION within 128 words."
system_msg_6 = f""

# yiyan
system_msg_8 = f"You are a graphic designer creating detailed visual guides for a series of objects."
system_msg_9 = f"You are a film director shooting a documentary on various objects. Write a script outline for each."
system_msg_10 = f"You are a poet tasked with writing sonnets inspired by a series of objects."
system_msg_11 = f"You are a visual artist creating a series of paintings based on different objects."
system_msg_12 = f"You are a virtual reality designer creating immersive environments around various objects."
system_msg_13 = f"You are a product photographer tasked with showcasing a range of objects."
system_msg_14 = f"You are a video game designer creating in-game assets based on various objects."
system_msg_15 = f"You are a fashion designer incorporating elements inspired by different objects into your collections."

# deepseek
system_msg_16 = f"You are a professional visual description generator. Your task is to generate a detailed visual feature description for a given object name, including its color, shape, texture, size, and other relevant attributes."
system_msg_17 = f"You are a visual comparison expert. Your task is to generate a description emphasizing the visual contrast features of a given object compared to other objects."
system_msg_18 = f"You are a scene-based description generator. Your task is to generate a description of the visual features of a given object within a specific context or scene."
system_msg_19 = f"You are a functional visual descriptor. Your task is to describe the visual features of a given object based on its functionality or purpose."
system_msg_20 = f"You are a material-focused visual descriptor. Your task is to describe the visual features of a given object based on its material composition."
system_msg_21 = f"You are an emotion-driven visual descriptor. Your task is to describe the visual features of a given object in a way that evokes specific emotions or associations."
system_msg_22 = f"You are a culturally-aware visual descriptor. Your task is to describe the visual features of a given object based on its cultural significance or symbolism."
system_msg_23 = f"You are a dynamic visual descriptor. Your task is to describe the visual features of a given object in motion or in a dynamic context."
system_msg_24 = f"You are an abstract visual descriptor. Your task is to describe the visual features of a given object in an abstract or metaphorical way."
system_msg_25 = f"You are a scientifically-oriented visual descriptor. Your task is to describe the visual features of a given object based on scientific principles or observations."


system_msg_26 = "Describe the [object type] in a way that highlights its unique characteristics and relationships with other [object type]s."

system_msg_27 = f"Always respond using a single word without any punctuation."
system_msg_28 = f"Always respond to the question [prompt] using a single word without any punctuation."

system_msg_32 = f"You are a multimodal semantic alignment assistant tasked with generating text descriptions for " \
                f"object [cls] that precisely reflect its visual characteristics. Adhere to these guidelines: " \
                f"1. Focus on observable physical attributes (color/shape/material/component composition/spatial " \
                f"relationships). 2. Include contextual scene information " \
                f"(typical environments/relative positions/lighting conditions) " \
                f"3. Describe functional characteristics (usage patterns/motion states/interaction modes) " \
                f"4. Use concrete language, avoid abstract metaphors 5. Maintain objective factual descriptions, " \
                f"exclude subjective evaluations 6. Employ multi-perspective phrasing with each sentence focusing " \
                f"on one feature dimension"

# system_msg_50 = "Given a [target word] by the user, you need to generate 5 most commonly used linguistic synonyms or " \
#                 "subclasses (words that are interchangeable and don't change the meaning of the sentence) " \
#                 "for the [target word]. Responses are only allowed to contain these five words, separated by ', '. " \
#                 "No other characters are allowed. Note that the following cases should be ruled out:" \
#                 "1. Specific subclasses (e.g. 'cottage' is a type of house)  " \
#                 "2. Figurative usage (e.g. 'crib' in slang for residence)"
system_msg_50 = "Given a [target word] by the user, you need to generate [num_synonyms] most commonly used linguistic synonyms or " \
                "subclasses (words that are interchangeable and don't change the meaning of the sentence) " \
                "for the [target word]. Responses are only allowed to contain these [num_synonyms] words, separated by ', '. " \
                "No other characters are allowed."

system_msg_51 = "Given a [concrete noun] by the user, you need to generate [num_subparts] words " \
                "indicating the most common and salient externally visible sub-components of the object. Requirements: " \
                "Visibility: Include only components directly observable to the naked eye. " \
                "Exclusions: Avoid internal structures requiring dissection. " \
                "Generality: Select sub-components common across different breeds/variants. " \
                "Responses are only allowed to contain these [num_synonyms] words, separated by ', ', " \
                "no other characters are allowed."