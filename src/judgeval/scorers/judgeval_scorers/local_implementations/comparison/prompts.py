

class ComparisonTemplate:

    @staticmethod
    def find_differences(criteria: str, description: str, actual_output: str, expected_output: str):
        return f"""==== TASK INSTRUCTIONS ====
You will be presented with two documents, an actual_output and a expected_output. The expected_output is superior to the actual_output in many qualities, but we are interested in finding the differences between these documents on the basis of {criteria}.

We define {criteria} to be the {description}.

Between the two documents, identify all differences between them on the basis of {criteria} if there is enough of a difference between the two. When you identify all the differences, please mark the specific sentences that are affected by these differences. Return a JSON containing a list of objects, each of which have the structure:
{{
    "actual_output_sentence": <quoted sentence from actual output>,
    "expected_output_sentence": <quoted sentence from expected output>,
    "reason": <reason for difference>
}}

It is okay if you dont think there is enough of a difference between the two documents to add to the list.

==== FORMATTING YOUR ANSWER ====
Please return your answer in JSON format, with the "differences" key as a list of JSON objects. Each JSON object should have 3 fields: 'actual_output_sentence', 'expected_output_sentence', and 'reason'. 
The 'difference' key should be a string representing the difference between the two documents if there is one. 
The 'reason' key should be a string explaining why the difference is important.

==== EXAMPLES 1====

Final Draft: RE: Recommendation Letter on behalf of Mr. Alex Johnson

Dear USCIS Officer:

I am writing this letter to express my strong support for Mr. Alex Johnson and his petition for an O-1 visa. As a highly accomplished entrepreneur and technology engineer, Mr. Johnson has demonstrated extraordinary ability in the field of time series analysis and innovative technology applications, achieving notable success in his field.

As the founder of Venture Partners and Capital Innovations, two prominent venture capital firms, I have been actively involved in the technology and entrepreneurship ecosystems for over three decades. My investments have helped launch and scale numerous successful companies, including StreamTech, OpenAI, FutureCorp, and BlockChain Solutions. These firms have transformed their respective industries and achieved significant market impact.

My educational background includes a Bachelor of Science in Electrical Engineering from Columbia University and an MBA from Yale School of Management. This combination of technical and business expertise has been instrumental in my ability to identify, fund, and guide promising technology ventures. In addition to my investment activities, I have also been a strong advocate for entrepreneurial education. In 2012, I founded Innovation Academy, an immersive entrepreneurship program that has trained over 1,000 students from more than 60 countries. The program equips participants with the skills, mindset, and network needed to succeed as entrepreneurs in today's fast-paced, technology-driven world.

Throughout my career, I have been recognized for my contributions to the venture capital and entrepreneurship communities. I have been honored as "Global Leader in Innovation" by the International Entrepreneurship Forum and have been featured on the Entrepreneur's "Top Investors" list multiple times. These accolades reflect my track record of identifying and supporting groundbreaking technologies and visionary entrepreneurs.

As an author and thought leader, I have shared my insights on entrepreneurship and innovation through my book, "Becoming the Innovation Leader," as well as through numerous speaking engagements and media appearances. I remain committed to fostering the next generation of entrepreneurs and promoting the transformative potential of technology.

It is through my role at Innovation Academy that I first became acquainted with Mr. Alex Johnson, who was selected to participate in the GET x Innovation Training program in the summer of 2023, a partnership between Innovation Academy and the Global Entrepreneurs Network. This program aims to foster entrepreneurial talent and support innovative startups from around the globe. The GET x Innovation Training program follows a rigorous selection process, selecting only a small number of teams each year from hundreds of applications. Mr. Johnson successfully navigated through three rounds of interviews to secure a place in this prestigious program located in San Mateo, California.

Participants engage in an intensive curriculum designed to accelerate their entrepreneurial journey and equip them with the skills needed to succeed in the global startup ecosystem. The curriculum covers essential aspects such as developing clear business visions, discovering financial resources, learning business techniques, and mastering negotiation and communication skills.

Throughout the program, participants have the opportunity to learn from experienced mentors, attend workshops, and network with investors and industry leaders. They are challenged to refine their business ideas, develop their leadership skills, and prepare to launch and scale their ventures. Mr. Johnson's participation in the GET x Innovation Training program is a testament to his entrepreneurial drive and the potential of his startup venture, TechGenius. Being selected for this competitive program required demonstrating a compelling vision, technical aptitude, and the ability to thrive in a high-pressure environment. During his time in the program, Mr. Johnson actively engaged in the curriculum, seeking guidance from mentors and collaborating with his peers. He demonstrated a keen ability to absorb new knowledge and apply it to his venture. His commitment to personal growth and his startup's success was evident throughout the program. As an alumnus of the GET x Innovation Training program, Mr. Johnson has joined a global network of entrepreneurs who have gone through the transformative experience of Innovation Academy's training. This network provides ongoing support, resources, and opportunities to help alumni continue to grow and scale their businesses.

Having overseen the participation of hundreds of students in various programs at Innovation Academy, I can wholeheartedly attest that Mr. Johnson stands out as one of the most extraordinary individuals that I have had the pleasure of mentoring. His passion, resilience, and insightful approach to problem-solving set him apart from his peers. Throughout the program, Mr. Johnson showed an outstanding ability to absorb complex concepts and apply them creatively to his venture. He consistently demonstrated a level of dedication and strategic thinking that is rare even among the most driven entrepreneurs. It is this remarkable blend of skills and determination that has enabled him to achieve significant milestones and continue to excel in his endeavors.

Notably, Mr. Johnson's extraordinary abilities and potential were recognized through significant investments and selective program acceptances. For example, Mr. Johnson secured an acceptance to the NexGen Tech Accelerator for his venture, TechGenius, shortly after completing the Innovation Training program. TechGenius is a pioneering platform that leverages technology to enhance business operations and decision-making. It provides tools that automate routine tasks, analyze data for actionable insights, and improve customer interactions through innovative solutions. By integrating advanced analytics and machine learning, TechGenius streamlines workflows and optimizes productivity for businesses of all sizes. The NexGen Tech Accelerator, a program launched by Visionary Ventures in collaboration with the Global Technology Innovation Authority and the National Development Program, aims to foster the growth of promising tech startups in emerging markets. This 10-week program provides selected participants with access to capital, resources, and mentorship to help them scale their ventures.

The NexGen Tech Accelerator is renowned for its rigorous and highly selective admission process, accepting only a small percentage of applicants each cycle. This elite program is designed to identify and nurture the most promising tech startups, providing them with invaluable resources and support. Successful candidates gain access to a substantial seed investment, expert mentorship, state-of-the-art facilities, and a rich network of industry leaders, thus positioning participants for accelerated growth and success in the competitive tech landscape.

Mr. Johnson's acceptance into the NexGen Tech Accelerator is a significant achievement, as it demonstrates the potential and quality of his startup in the field of technology innovation. His selection into the NexGen Tech Accelerator is a testament to his abilities as an entrepreneur and the strength of his startup's vision. It positions him among a select group of individuals who are driving innovation and shaping the future of technology in the region and beyond.

In conjunction with his acceptance into the NexGen Tech Accelerator, Mr. Johnson also secured a substantial pre-seed investment for his startup, TechGenius. This significant funding, procured in a highly competitive tech startup landscape, is a testament to the high confidence investors have in Mr. Johnson’s innovative capabilities and the immense potential of his venture. This remarkable achievement not only highlights Mr. Johnson’s proficiency in developing groundbreaking technology solutions but also underscores his adeptness at effectively communicating his vision and strategy to expert investors. His ability to secure such a coveted investment illustrates his exceptional skill in navigating the complex world of venture funding, a feat that sets him apart from his peers.

Such financial backing from Visionary Ventures is indicative of Mr. Johnson's strong business acumen and his ability to inspire confidence among seasoned investors. It speaks volumes about his convincing pitch, the robustness of his business model, and the forward-thinking nature of TechGenius. The investment enables him to further develop and enhance the platform’s tools, which aim to revolutionize business operations and decision-making.

By securing both the NexGen Tech Accelerator acceptance and the significant investment, Mr. Johnson has positioned TechGenius for accelerated growth and success. These milestones provide the financial resources and expert mentorship necessary for scaling his venture to new heights. Mr. Johnson's achievements reflect his extraordinary talent and promise in the field of technology innovation, marking him as a visionary entrepreneur poised to make significant contributions to the industry.

Building on the success of TechGenius, Mr. Johnson has now embarked on an exciting new venture as the Founder and CEO of SmartTech, where he is instrumental in the development and strategic growth of the company. Specializing as a technology engineer in the fields of time series analysis and innovative applications, Mr. Johnson leverages advanced technology for pioneering applications in various sectors. His responsibilities span critical areas, including business model iteration, funding acquisition, and customer engagement—each essential to the company's success and market positioning.

Mr. Johnson has diligently led the ongoing process of iterating and refining SmartTech's business model to align with market needs and industry trends. This involves rigorous analysis of market feedback and technological advancements. By continuously testing and optimizing various business strategies, Mr. Johnson has ensured that the company's offerings are both viable and scalable.

In addition, Mr. Johnson has played a pivotal role in customer acquisition and establishing strategic partnerships that drive the company's growth. A notable accomplishment is securing a signed Non-Disclosure Agreement (NDA) with HealthCorp, a leading provider of healthcare technology solutions. This agreement is a testament to the trust and confidence that major industry players have in SmartTech’s capabilities. The collaboration with HealthCorp not only validates their technology but also opens doors to further opportunities in the tech sector, positioning SmartTech as a credible player in the technology landscape.

Further bolstering the promise of SmartTech, Mr. Johnson and his venture were recently selected for the prestigious Founder in Residence program in New York City. This program is designed to support entrepreneurs from inception to success, offering invaluable support that includes co-founder matching, comprehensive business model validation, initial capital, expansion support, and follow-on funding. Mr. Johnson’s inclusion in this program further underscores his potential to drive groundbreaking advancements in technology.

In conclusion, Mr. Johnson's journey highlights his impressive capabilities and unwavering dedication to innovation. From securing significant pre-seed investment from Visionary Ventures for TechGenius, to being selected for the highly competitive NexGen Tech Accelerator, and now leading SmartTech as a pioneering force in technology, Mr. Johnson has consistently demonstrated his remarkable talent and vision. His expertise as a technology engineer specializing in innovative applications has enabled him to develop cutting-edge solutions poised to make significant contributions to various industries. Moreover, his recent inclusion in the Founder in Residence program further underscores his potential to drive advancements in technology.

As someone who has been recognized for leadership in entrepreneurship and has witnessed the evolution of many startups, I can confidently say that Mr. Johnson’s accomplishments are truly commendable. From the moment I met him at the Innovation Training program, I knew he would make significant strides in the field, as he has done. His ability to navigate complex challenges, secure critical funding, and establish strategic partnerships demonstrates a level of maturity and insight that sets him apart from his peers. With a proven track record of success and a commitment to continuous growth, Mr. Johnson is well-positioned to shape the future of technology. He is undoubtedly a visionary entrepreneur whose work will leave a lasting impact on the industry, driving innovation and setting new standards for excellence.

Yours truly,



Chris Anderson
Founder at Venture Partners/Capital Innovations/Innovation Academy
chris@venture.vc

Rough Draft:
September 23, 2024

United States Citizenship and Immigration Service Regional Service Center

RE: Letter of Support for Alex Johnson's O-1 Visa Petition

Dear USCIS Officer:

I am writing this letter to express my strong support for Alex Johnson's petition for an O-1 visa. As a highly accomplished entrepreneur and technologist, Alex has demonstrated extraordinary ability and achieved notable success in his field at a young age.

[Recommender Background]

As the founder of Venture Partners and Capital Innovations, two prominent venture capital firms, I have been actively involved in the technology and entrepreneurship ecosystems for over three decades. My investments have helped launch and scale numerous successful companies, including StreamTech, OpenAI, FutureCorp, and BlockChain Solutions. These firms have transformed their respective industries and achieved significant market impact.

My educational background includes a Bachelor of Science in Electrical Engineering from Columbia University and an MBA from Yale School of Management. This combination of technical and business expertise has been instrumental in my ability to identify, fund, and guide promising technology ventures.

In addition to my investment activities, I have also been a strong advocate for entrepreneurial education. In 2012, I founded Innovation Academy, an immersive entrepreneurship program that has trained over 1,000 students from more than 60 countries. The program equips participants with the skills, mindset, and network needed to succeed as entrepreneurs in today's fast-paced, technology-driven world.

Throughout my career, I have been recognized for my contributions to the venture capital and entrepreneurship communities. I have been honored as "Global Leader in Innovation" by the International Entrepreneurship Forum and have been featured on the Entrepreneur's "Top Investors" list multiple times. These accolades reflect my track record of identifying and supporting groundbreaking technologies and visionary entrepreneurs.

As an author and thought leader, I have shared my insights on entrepreneurship and innovation through my book, "Becoming the Innovation Leader," as well as through numerous speaking engagements and media appearances. I remain committed to fostering the next generation of entrepreneurs and promoting the transformative potential of technology.

[Innovation Academy and Global Entrepreneur Training Program]

It is through my role at Innovation Academy that I first became acquainted with Alex Johnson. Alex was selected to participate in the GET x Innovation Training program, a partnership between Innovation Academy and the Global Entrepreneurs Network. This program aims to foster entrepreneurial talent and support innovative startups from around the globe.

The GET x Innovation Training program follows a rigorous selection process, choosing only a small number of teams each year to join the program in San Mateo, California. Participants engage in an intensive curriculum designed to accelerate their entrepreneurial journey and equip them with the skills needed to succeed in the global startup ecosystem.

Throughout the program, participants have the opportunity to learn from experienced mentors, attend workshops and speaker sessions, and network with investors and industry leaders. They are challenged to refine their business ideas, develop their leadership skills, and prepare to launch and scale their ventures.

Alex's participation in the GET x Innovation Training program is a testament to his entrepreneurial drive and the potential of his startup. Being selected for this competitive program required demonstrating a compelling vision, technical aptitude, and the ability to thrive in a fast-paced, high-pressure environment.

During his time in the program, Alex actively engaged in the curriculum, seeking guidance from mentors and collaborating with his peers. He demonstrated a keen ability to absorb new knowledge and apply it to his venture. His commitment to personal growth and his startup's success was evident throughout the program.

As an alumnus of the GET x Innovation Training program, Alex has joined a global network of entrepreneurs who have gone through the transformative experience of Innovation Academy's training. This network provides ongoing support, resources, and opportunities to help alumni continue to grow and scale their businesses.

[Alex Johnson's Acceptance Into GET x Innovation Training Program]

Alex Johnson's acceptance into the GET x Innovation Training program is a significant achievement that demonstrates his potential in the field of entrepreneurship and technology. The GET x Innovation Training program is a highly competitive initiative that selects a small number of promising teams each year to participate in an intensive training program at Innovation Academy in San Mateo, California.

The selection process for the GET x Innovation Training program is rigorous, with hundreds of applicants vying for a limited number of spots. Teams are chosen based on the strength of their business ideas, the technical capabilities of their founders, and their potential to create significant impact in their respective industries. Alex's selection into this program is a testament to the quality of his startup concept and his abilities as an entrepreneur.

The GET x Innovation Training program provides participants with a comprehensive curriculum designed to accelerate their entrepreneurial journey. The program covers a wide range of topics essential for startup success, including business model development, customer acquisition, fundraising, and leadership skills. Participants engage in hands-on workshops, mentoring sessions with industry experts, and networking opportunities with investors and fellow entrepreneurs.

Throughout the program, participants work to refine their business ideas, build out their products, and develop go-to-market strategies. They receive guidance and support from experienced mentors who help them navigate the challenges of building and scaling a successful startup. The program culminates in a demo day event where participants pitch their ventures to a panel of investors and industry leaders.

By completing the GET x Innovation Training program, Alex has gained valuable skills and knowledge that will benefit him as he continues to grow his startup. The program's emphasis on practical, real-world learning experiences has equipped him with the tools and insights needed to succeed in the fast-paced world of technology entrepreneurship. Additionally, the network of mentors, investors, and fellow entrepreneurs he has built through the program will serve as a valuable resource as he scales his business.

[Membership Criterion: NexGen Tech Accelerator]

The NexGen Tech Accelerator, a program launched by Visionary Ventures in collaboration with the Global Technology Innovation Authority and the National Development Program, aims to foster the growth of promising tech startups in emerging markets. This 10-week program provides selected participants with access to capital, resources, and mentorship to help them scale their ventures.

NexGen follows a highly selective application process, ensuring that only the most innovative and promising tech projects are chosen to participate. The program is structured into three distinct phases: pre-program, program, and post-program, each designed to provide tailored support to the startups at different stages of their development.

Alex Johnson's acceptance into the NexGen Tech Accelerator is a significant achievement, as it demonstrates the potential and quality of his startup in the field of technology innovation. The selection process for NexGen is competitive, with a limited number of spots available in each cohort. Alex's participation in the program provides him with access to invaluable resources, including expert mentorship, workshops, and networking opportunities with investors and industry leaders.

As a participant in NexGen, Alex has the opportunity to refine his business model, develop his product, and create effective go-to-market strategies. The program's focus on practical, hands-on learning experiences equips participants with the skills and knowledge needed to navigate the challenges of building and scaling a successful tech startup.

By being part of the NexGen Tech Accelerator, Alex joins a community of like-minded entrepreneurs and innovators who are at the forefront of the technology revolution. This network can serve as a valuable resource for collaboration, support, and potential partnerships as Alex continues to grow his venture.

Alex's selection into the NexGen Tech Accelerator is a testament to his abilities as an entrepreneur and the strength of his startup's vision. It positions him among a select group of individuals who are driving innovation and shaping the future of technology in the region and beyond.

[Awards Criterion]

Alex Johnson's exceptional abilities and potential have been recognized through significant investments and selective program acceptances. Most notably, Alex secured a $100,000 pre-seed investment from Visionary Ventures for his startup, TechGenius. Pre-seed funding, which typically ranges from $50,000 to $250,000, is the earliest stage of venture capital financing. Securing an investment of this size at such an early stage is a testament to the strength of Alex's business concept and his ability to convince investors of its potential.

In addition to the Visionary Ventures investment, Alex was accepted into the highly competitive NexGen Tech Accelerator program. NexGen aims to support and scale promising tech startups in emerging markets.

Acceptance into the NexGen Tech Accelerator is a significant achievement, as the program follows a rigorous selection process to identify the most innovative and promising tech projects. As part of the program, Alex's startup will receive access to capital, resources, and mentorship to help accelerate its growth. The fact that Alex's venture was chosen to participate in this program is a clear indication of its potential and Alex's abilities as an entrepreneur.

Furthermore, Alex has demonstrated his technical prowess and problem-solving skills through his performance in hackathons and coding competitions. He and his team won first place out of hundreds of teams in the Global Tech Challenge, a testament to their ability to develop innovative solutions using cutting-edge technology. Alex also received the Best Innovator Award at the FutureTech Hackathon event, further showcasing his skills and creativity.

These investments, program acceptances, and competition wins serve as clear evidence of Alex Johnson's extraordinary ability and potential in the field of technology and entrepreneurship. They demonstrate his capacity to innovate, build compelling products, and attract the support of investors and industry leaders. As Alex continues to grow his venture and make strides in the tech space, these achievements will undoubtedly serve as important milestones in his journey as a technology entrepreneur.

[Conclusion]

In conclusion, Alex Johnson has demonstrated a track record of entrepreneurial achievement and potential that distinguishes him as an individual of extraordinary ability. His participation in the highly selective GET x Innovation Training program, acceptance into the NexGen Tech Accelerator, and the significant pre-seed investment secured from Visionary Ventures all serve as testament to his capabilities and promise in the field of technology entrepreneurship. I believe Alex possesses the drive, skills, and vision to make notable contributions to the startup ecosystem in the United States.


Criteria: Use of Evidence

Description: claims should be substantiated with evidence. Facts should be used to emphasize the exceptional nature of the individual. Flowery, verbose compliments to the individual are encouraged as long as claims are filled with specific examples.

Example JSON:

{{
  "differences": [
    {{
      "actual_output_sentence": "Alex Johnson's acceptance into the GET x Innovation Training program is a significant achievement that demonstrates his potential in the field of entrepreneurship and technology.",
      "expected_output_sentence": "Mr. Johnson successfully navigated through three rounds of interviews to secure a place in this prestigious program located in San Mateo, California.",
      "reason": "Final draft adds specific evidence about the competitive selection process (three rounds of interviews) to substantiate the claim of rigor."
    }},
    {{
      "actual_output_sentence": "Alex secured a $100,000 pre-seed investment from Visionary Ventures for his startup, TechGenius.",
      "expected_output_sentence": "Mr. Johnson also secured a substantial pre-seed investment for his startup, TechGenius. This significant funding, procured in a highly competitive tech startup landscape, is a testament to the high confidence investors have in Mr. Johnson’s innovative capabilities...",
      "reason": "Final draft removes specific dollar amounts but adds contextual evidence about the competitive landscape and investor confidence to emphasize significance."
    }},
    {{
      "actual_output_sentence": "TechGenius is a pioneering platform that leverages technology to enhance business operations and decision-making.",
      "expected_output_sentence": "TechGenius provides tools that automate routine tasks, analyze data for actionable insights, and improve customer interactions through innovative solutions. By integrating advanced analytics and machine learning, TechGenius streamlines workflows...",
      "reason": "Final draft adds technical specifics (advanced analytics, machine learning, workflow automation) as concrete evidence of innovation."
    }},
    {{
      "actual_output_sentence": "Alex and his team won first place out of hundreds of teams in the Global Tech Challenge.",
      "expected_output_sentence": "A notable accomplishment is securing a signed Non-Disclosure Agreement (NDA) with HealthCorp...",
      "reason": "Final draft replaces competition wins with partnership evidence (HealthCorp NDA) as more relevant professional validation."
    }},
    {{
      "actual_output_sentence": "NexGen follows a highly selective application process... structured into three distinct phases: pre-program, program, and post-program.",
      "expected_output_sentence": "The NexGen Tech Accelerator is renowned for its rigorous and highly selective admission process, accepting only a small percentage of applicants each cycle. This elite program... provides them with invaluable resources and support.",
      "reason": "Final draft strengthens evidence by quantifying selectivity ('small percentage of applicants') and specifying resource advantages."
    }},
    {{
      "actual_output_sentence": "Alex has demonstrated his technical prowess through his performance in hackathons.",
      "expected_output_sentence": "Specializing as a technology engineer in the fields of time series analysis and innovative applications, Mr. Johnson leverages advanced technology for pioneering applications in various sectors.",
      "reason": "Final draft replaces hackathon mentions with specific technical domain expertise (time series analysis) as stronger professional evidence."
    }},
    {{
      "actual_output_sentence": "The program culminates in a demo day event where participants pitch their ventures to investors.",
      "expected_output_sentence": "Mr. Johnson has positioned TechGenius for accelerated growth and success. These milestones provide the financial resources and expert mentorship necessary for scaling his venture to new heights.",
      "reason": "Final draft adds outcome-focused evidence (positioning for growth, mentorship access) rather than procedural demo day details."
    }},
    {{
      "actual_output_sentence": "Alex's venture was chosen to participate in this program is a clear indication of its potential.",
      "expected_output_sentence": "Successful candidates gain access to a substantial seed investment, expert mentorship, state-of-the-art facilities, and a rich network of industry leaders...",
      "reason": "Final draft substantiates program value with specific resource evidence (seed investment, facilities, network)."
    }}
  ]
}}

===========
YOUR TURN

Expected_output: {expected_output}
Actual_output: {actual_output}
Criteria: {criteria}
Description: {description}

Differences based on criteria:
"""