from typing import Any, Dict, List, Tuple, TypeAlias
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, BoundedLinearExpression
from pandas import DataFrame

StudentPreferences: TypeAlias = Dict[str, List[str]]
CourseMaxStudents: TypeAlias = Dict[str, int]


def get_example_problem():
    student_preferences: StudentPreferences = {
        'alice': ['alchemy', 'sorcery', 'arithmancy'],
        'bob': ['alchemy', 'sorcery', 'magical beasts'],
        'mike truck': ['alchemy', 'quidditch', 'sorcery'],
        'bobson dugnutt': ['alchemy', 'quidditch', 'magical beasts']
    }
    course_max_students: CourseMaxStudents = {
        'alchemy': 2,
        'sorcery': 1,
        'arithmancy': 10,
        'magical beasts': 30,
        'quidditch': 30
    }
    return student_preferences, course_max_students


class CourseAssignmentVariables:
    def __init__(self, initial_variables: List[Tuple[str, str, cp_model.IntVar]]):
        self.variables: DataFrame = DataFrame(
            data=initial_variables,
            columns=['student', 'course', 'variable']
        )

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        # not sure how to meaningfully compare the IntVar objects, so dropping those for now
        return self.variables[['student', 'course']].equals(other.variables[['student', 'course']])

    def get_by_student_name(self, name: str) -> List[cp_model.IntVar]:
        student_vars: List[cp_model.IntVar] = self.variables.query(f"student == '{name}'")['variable'].to_list()
        return student_vars


def generate_course_assignment_variables(students: StudentPreferences, courses: CourseMaxStudents, model: CpModel) -> CourseAssignmentVariables:
    student_names: List[str] = list(students.keys())
    course_names: List[str] = list(courses.keys())
    initial_variables: List[Tuple[str, str, cp_model.IntVar]] = [
        (student_name, course_name, model.NewIntVar(0, 1, f'{student_name} in {course_name}'))
        for student_name in student_names for course_name in course_names
    ]
    assignments = CourseAssignmentVariables(initial_variables)
    return assignments


def main():
    student_preferences: StudentPreferences
    course_max_students: CourseMaxStudents
    student_preferences, course_max_students = get_example_problem()
    model = cp_model.CpModel()
    assignment_variables: CourseAssignmentVariables = generate_course_assignment_variables(
        student_preferences,
        course_max_students,
        model
    )

    student_names: list = list(student_preferences.keys())
    exactly_one_course_constraints: List[BoundedLinearExpression] = []
    for student_name in student_names:
        variables_for_student: List[cp_model.IntVar] = assignment_variables.get_by_student_name(student_name)
        constraint: BoundedLinearExpression = sum(variables_for_student) == 1
        exactly_one_course_constraints.append(constraint)

    # for c in constraints:
    #     model.Add(c)


    exactly_one_course_constraints = generate_each_student_exactly_one_course_constraints(
        student_preferences,
        course_max_students,
        assignment_variables,
        model
    )
    model.Add(*exactly_one_course_constraints)

    no_more_than_max_students_in_course_constraints = generate_max_students_in_course_constraints(
        student_preferences,
        course_max_students,
        assignment_variables,
        model
    )
    model.Add(*no_more_than_max_students_in_course_constraints)

    only_preferred_courses_constraints = generate_only_preferred_courses_constraints(
        student_preferences,
        course_max_students,
        assignment_variables,
        model
    )
    model.Add(*only_preferred_courses_constraints)

    total_cost = make_total_cost_expression(
        student_preferences,
        course_max_students,
        assignment_variables,
        model
    )

    model.Minimize(total_cost)

    # see for example: https://developers.google.com/optimization/cp/cp_example
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
        print(f'x = {solver.Value(x)}')
        print(f'y = {solver.Value(y)}')
        print(f'z = {solver.Value(z)}')
    else:
        print('No solution found.')
